class ShadowRenderer
	{
	public:
		ShadowRenderer(ResourceManager* rm, TempRingUniformBuffer& uniformRingBuffer) : m_uniformRingBuffer(uniformRingBuffer)
		// TODO: Use GPU temp allocator instead of pre-allocated uniformRingBuffer. Requires BindGroup update.
		{
			auto timer = ScopedTimer("ShadowRenderer init");

			List<uint8> shadowShaderVS = ReadFile("../data/pack/gen/shader/hyper/mesh_shadow_vert.spv");

			// Render pass
			m_renderPassLayout = rm->createRenderPassLayout({
				.depthTargetFormat = FORMAT::D32_FLOAT,
				.subpasses = { {.depthTarget = true} } });

			m_renderPass = rm->createRenderPass({
				.layout = m_renderPassLayout,
				.depthTarget = {.nextUsage = TEXTURE_LAYOUT::SAMPLED, .clearZ = 0.0f} // inverse Z for better quality
				});

			// Bindings layouts
			m_globalsBindingsLayout = rm->createBindGroupLayout({
				.buffers = {
					{.slot = 0} // Global uniforms (camera matrices, etc)
				} });

			m_drawBindingsLayout = ResourceManager::ptr->createBindGroupLayout(BindGroupLayoutDesc{
				.buffers = {
					{.slot = 0, .type = BindGroupLayoutDesc::BufferBinding::TYPE::UNIFORM_DYNAMIC_OFFSET, .stages = BindGroupLayoutDesc::STAGE_FLAGS::VERTEX} // Transform matrices
				} });

			// Shader
			m_shader = rm->createShader({
				.VS {.byteCode = shadowShaderVS, .entryFunc = "main"},
				.bindGroups = {
					{ m_globalsBindingsLayout }, // Globals bind group (0)
					{ },  // Not used (1)
					{ }, // Not used (2)
					{ m_drawBindingsLayout } // Per draw bind group (3)
				},
				.graphicsState = {
					.depthTest = COMPARE::GREATER_OR_EQUAL, // inverse Z for better quality
					.vertexBufferBindings {
						{
							// Position vertex buffer (0)
							.byteStride = 12, .attributes = {
								{.byteOffset = 0,.format = FORMAT::RGB32_FLOAT}
							}
						}
					},
					.renderPassLayout = m_renderPassLayout
				} });

			// Shadow map texture
			auto shadowDimensions = Vector3I(4096, 4096, 1);

			m_shadowMap = rm->createTexture({
				.dimensions = shadowDimensions,
				.mips = 1,
				.format = FORMAT::D32_FLOAT,
				.usage = TextureDesc::USAGE_DEPTH_STENCIL | TextureDesc::USAGE_SAMPLED,
				.sampler = {.compare = COMPARE::LESS_OR_EQUAL} });

			m_framebuffer = rm->createFramebuffer({
				.dimensions = shadowDimensions,
				.renderPassLayout = m_renderPassLayout,
				.depthTarget = m_shadowMap });

			// Uniform buffer
			m_globalUniforms = rm->createBuffer({ .byteSize = sizeof(ShadowGlobalUniforms) });
			m_uniforms = (ShadowGlobalUniforms*)rm->getBufferData(m_globalUniforms);

			// Bindings
			m_globalBindings = rm->createBindGroup({
			.layout = m_globalsBindingsLayout,
			.buffers = {{.buffer = m_globalUniforms}} });

			m_drawBindings = rm->createBindGroup({
				.layout = m_drawBindingsLayout,
				.buffers = {{.buffer = uniformRingBuffer.getBuffer(), .byteSize = (uint32)sizeof(ShadowDrawUniforms)}} });

		}

		void destroy(ResourceManager* rm)
		{
			auto timer = ScopedTimer("ShadowRenderer destroy");
			rm->deleteBindGroup(m_drawBindings);
			rm->deleteBindGroup(m_globalBindings);
			rm->deleteBuffer(m_globalUniforms);
			rm->deleteFramebuffer(m_framebuffer);
			rm->deleteTexture(m_shadowMap);
			rm->deleteShader(m_shader);
			rm->deleteBindGroupLayout(m_drawBindingsLayout);
			rm->deleteBindGroupLayout(m_globalsBindingsLayout);
			rm->deleteRenderPass(m_renderPass);
			rm->deleteRenderPassLayout(m_renderPassLayout);
		}

		void render(Span<const SceneObject> sceneObjects, const SunLight& light)
		{
			// Update uniforms
			// TODO/FIXME: Needs to be double buffered!
			m_uniforms->viewProj = light.viewProj;

			CommandBuffer* commandBuffer = Renderer::ptr->beginCommandRecording(COMMAND_BUFFER_TYPE::OFFSCREEN);
			RenderPassRenderer* passRenderer = commandBuffer->beginRenderPass(m_renderPass, m_framebuffer);

			// Draws
			// TODO: Implement culling
			List<Draw> draws((uint32)sceneObjects.size);
			for (const SceneObject& sceneObject : sceneObjects)
			{
				auto alloc = m_uniformRingBuffer.bumpAlloc<ShadowDrawUniforms>();
				alloc.ptr->model = Matrix3x4::translate(sceneObject.position);

				draws.insert({
					.shader = m_shader,
					.mesh = sceneObject.mesh,
					.bindGroup1 = Handle<BindGroup>(),
					.dynamicBufferOffset0 = alloc.offset });
			}

			DrawArea drawArea{ .bindGroup0 = m_globalBindings, .bindGroupDynamicOffsetBuffers = m_drawBindings, .drawOffset = 0, .drawCount = (uint32)sceneObjects.size };
			passRenderer->drawSubpass(drawArea, draws);

			commandBuffer->endRenderPass(passRenderer);
			commandBuffer->submit();
		}

		Handle<Texture> getShadowMap() const { return m_shadowMap; }

	private:
		TempRingUniformBuffer& m_uniformRingBuffer;
		ShadowGlobalUniforms* m_uniforms;

		Handle<RenderPassLayout> m_renderPassLayout;
		Handle<RenderPass> m_renderPass;
		Handle<BindGroupLayout> m_globalsBindingsLayout;
		Handle<BindGroupLayout> m_drawBindingsLayout;
		Handle<Shader> m_shader;
		Handle<Texture> m_shadowMap;
		Handle<Framebuffer> m_framebuffer;
		Handle<Buffer> m_globalUniforms;
		Handle<BindGroup> m_globalBindings;
		Handle<BindGroup> m_drawBindings;
	};