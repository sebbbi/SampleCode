struct SunLight
{
    Vector3 direction;
    Matrix4x4 viewProj;

    void setDirection(Vector3 _direction)
    {
        direction = _direction;
        auto view = Matrix3x4::view(Vector3(0.0f, 0.0f, 0.0f), direction, Vector3(0.0f, 1.0f, 0.0f));
        auto proj = Matrix4x4::ortho(-80.0f, 80.0f, -80.0f, 80.0f, -80.0f, 80.0f);
        viewProj = Matrix4x4::mul(Matrix4x4::from3x4(view), proj);
    }
};

struct Camera
{
    Vector3 position;

    Vector3 direction;
    Matrix3x4 view;
    Matrix4x4 proj;
    Matrix4x4 viewProj;
    Matrix3x3 viewInvTranspose;

    void setPositionAspect(Vector3 _position, float aspect)
    {
        position = _position;
        view = Matrix3x4::view(Vector3(0.0f, 0.0f, 0.0f), _position, Vector3(0.0f, 1.0f, 0.0f));
        proj = Matrix4x4::projection(F_PI * 0.5f, aspect, 0.1f, 1000.0f);
        viewProj = Matrix4x4::mul(Matrix4x4::from3x4(view), proj);
        viewInvTranspose = Matrix3x3::inverse(Matrix3x3::from3x4(view));
    }
};

struct SceneObject
{
    Vector3 position;
    Handle<BindGroup> material;
    Handle<Mesh> mesh;
};

class Meshes
{
public:
    static constexpr uint32 NUM_MESHES = 1024;

    Meshes(ResourceManager* rm)
    {
        auto timerGenerateMesh = ScopedTimer("Meshes init");

        // Cube mesh
        static const uint32 numCubeFaces = 6;
        static const uint32 numIndices = numCubeFaces * 2 * 3;
        static const uint32 numVertices = numCubeFaces * 4;

        uint16 cubeFaceIndices[2 * 3 * 2] = {
            0, 2, 1, 2, 3, 1, // negative side
            1, 2, 0, 1, 3, 2, // positive side
        };

        Vector3 cubePositions[numVertices]{
            // X-
            Vector3(-1.0f, -1.0f, -1.0f),
            Vector3(-1.0f, 1.0f, -1.0f),
            Vector3(-1.0f , -1.0f, 1.0f),
            Vector3(-1.0f, 1.0f, 1.0f),

            // X+
            Vector3(1.0f, -1.0f, -1.0f),
            Vector3(1.0f, 1.0f, -1.0f),
            Vector3(1.0f , -1.0f, 1.0f),
            Vector3(1.0f, 1.0f, 1.0f),

            // Y-
            Vector3(-1.0f, -1.0f, -1.0f),
            Vector3(-1.0f, -1.0f, 1.0f),
            Vector3(1.0f, -1.0f, -1.0f),
            Vector3(1.0f, -1.0f, 1.0f),

            // Y+
            Vector3(-1.0f, 1.0f, -1.0f),
            Vector3(-1.0f, 1.0f, 1.0f),
            Vector3(1.0f, 1.0f, -1.0f),
            Vector3(1.0f, 1.0f, 1.0f),

            // Z-
            Vector3(-1.0f, -1.0f, -1.0f),
            Vector3(1.0f, -1.0f, -1.0f),
            Vector3(-1.0f, 1.0f, -1.0f),
            Vector3(1.0f, 1.0f, -1.0f),

            // Z+
            Vector3(-1.0f, -1.0f, 1.0f),
            Vector3(1.0f, -1.0f, 1.0f),
            Vector3(-1.0f, 1.0f, 1.0f),
            Vector3(1.0f, 1.0f, 1.0f),
        };

        Vector2 cubeFaceUVs[4]{
            Vector2(0.0f, 0.0f),
            Vector2(1.0f, 0.0f),
            Vector2(0.0f, 1.0f),
            Vector2(1.0f, 1.0f),
        };

        Vector3 cubeNormals[numCubeFaces]{
            Vector3(-1.0f, 0.0f, 0.0f), // X-
            Vector3(1.0f, 0.0f, 0.0f), // X+
            Vector3(0.0f, -1.0f, 0.0f), // Y-
            Vector3(0.0f, 1.0f, 0.0f), // Y+
            Vector3(0.0f, 0.0f, -1.0f), // Z-
            Vector3(0.0f, 0.0f, 1.0f), // Z+
        };

        Vector3 cubeTangents[numCubeFaces]{
            Vector3(0.0f, -1.0f, 0.0f), // X-
            Vector3(0.0f, 1.0f, 0.0f), // X+
            Vector3(0.0f, 0.0f, -1.0f), // Y-
            Vector3(0.0f, 0.0f, 1.0f), // Y+
            Vector3(-1.0f, 0.0f, 0.0f), // Z-
            Vector3(1.0f, 0.0f, 0.0f), // Z+
        };

        // Unique index/vertex buffer for each mesh
        // TODO/OPTIMIZE: Pack to one buffer instead and use binding .byteOffset
        for (uint32 i = 0; i < NUM_MESHES; i++)
        {
            m_indexBuffers[i] = rm->createBuffer({ .byteSize = sizeof(uint16) * numIndices, .usage = BufferDesc::USAGE_INDEX });
            uint16* indices = (uint16*)rm->getBufferData(m_indexBuffers[i]);
            for (uint32 j = 0; j < numIndices; j++)
            {
                uint32 face = j / (2 * 3);
                uint32 index = j % (2 * 3 * 2);
                indices[j] = cubeFaceIndices[index] + face * 4;
            }

            // Two vertex streams: position + properties. Faster for mobile TBDR GPUs and for shadows
            m_vertexBuffersPosition[i] = rm->createBuffer({ .byteSize = sizeof(Vector3) * numVertices, .usage = BufferDesc::USAGE_VERTEX });
            m_vertexBuffersProperties[i] = rm->createBuffer({ .byteSize = sizeof(VertexProperties) * numVertices, .usage = BufferDesc::USAGE_VERTEX });
            Vector3* positions = (Vector3*)rm->getBufferData(m_vertexBuffersPosition[i]);
            VertexProperties* properties = (VertexProperties*)rm->getBufferData(m_vertexBuffersProperties[i]);
            float randomMeshScale = 0.1f + frandom(1.4f);
            for (uint32 v = 0; v < numVertices; v++)
            {
                positions[v] = cubePositions[v] * randomMeshScale;
                properties[v].normal = Vector4toHalf4(Vector4(cubeNormals[v / 4], 0.0f));
                properties[v].tangent = Vector4toHalf4(Vector4(cubeTangents[v / 4], 1.0f));
                properties[v].color = { .r = (uint8)random(255), .g = (uint8)random(255), .b = (uint8)random(255), .a = (uint8)random(255) };
                properties[v].texcoord = Vector2toHalf2(cubeFaceUVs[v % 4]);
            }
        }

        // Meshes
        for (uint32 i = 0; i < NUM_MESHES; i++)
        {
            m_meshes[i] = rm->createMesh({
                .indexOffset = 0,
                .indexCount = numIndices,
                .vertexOffset = 0,
                .vertexCount = numVertices,
                .indexBuffer = m_indexBuffers[i],
                .vertexBuffers = {m_vertexBuffersPosition[i], m_vertexBuffersProperties[i], Handle<Buffer>()} });
        }
    }

    void destroy(ResourceManager* rm)
    {
        auto timer = ScopedTimer("Meshes destroy");
        for (uint32 i = 0; i < NUM_MESHES; i++) rm->deleteBuffer(m_indexBuffers[i]);
        for (uint32 i = 0; i < NUM_MESHES; i++) rm->deleteBuffer(m_vertexBuffersProperties[i]);
        for (uint32 i = 0; i < NUM_MESHES; i++) rm->deleteBuffer(m_vertexBuffersPosition[i]);
        for (uint32 i = 0; i < NUM_MESHES; i++) rm->deleteMesh(m_meshes[i]);
    }

    Handle<Mesh> getMesh(uint32 index) const { return m_meshes[index]; }

private:
    Handle<Buffer> m_vertexBuffersPosition[NUM_MESHES];
    Handle<Buffer> m_vertexBuffersProperties[NUM_MESHES];
    Handle<Buffer> m_indexBuffers[NUM_MESHES];
    Handle<Mesh> m_meshes[NUM_MESHES];
};

class Materials
{
public:
    static constexpr uint32 NUM_MATERIALS = 1024;

    Materials(ResourceManager* rm)
    {
        // Load textures
        auto timerLoadTextures = ScopedTimer("Materials - load textures");
        auto textureAlbedo = RGBA8ImageFile("../data/test/SD/RAW/tiles_albedo.png");
        auto textureNormal = RGBA8ImageFile("../data/test/SD/RAW/tiles_normal.png");
        auto textureProperties = RGBA8ImageFile("../data/test/SD/RAW/tiles_properties.png");
        timerLoadTextures.end();

        auto timerInitAssets = ScopedTimer("Materials init");

        // Upload textures to GPU
        m_materialTextureAlbedo = createTextureWithInitialData(rm, { .dimensions = Vector3I(textureAlbedo.header->width, textureAlbedo.header->height, 1), .format = FORMAT::RGBA8_UNORM }, textureAlbedo.texels);
        m_materialTextureNormal = createTextureWithInitialData(rm, { .dimensions = Vector3I(textureNormal.header->width, textureNormal.header->height, 1), .format = FORMAT::RGBA8_UNORM }, textureNormal.texels);
        m_materialTextureProperties = createTextureWithInitialData(rm, { .dimensions = Vector3I(textureProperties.header->width, textureProperties.header->height, 1), .format = FORMAT::RGBA8_UNORM }, textureProperties.texels);

        // Unique randomized uniform buffer for each material
        // TODO/OPTIMIZE: Pack to one buffer instead and use binding .byteOffset			
        for (uint32 i = 0; i < NUM_MATERIALS; i++)
        {
            m_materialUniforms[i] = rm->createBuffer({ .byteSize = sizeof(MaterialUniforms) });

            MaterialUniforms* matUniforms = (MaterialUniforms*)rm->getBufferData(m_materialUniforms[i]);
            auto color = Vector4(frandom(1.0), frandom(1.0), frandom(1.0), 1.0f);
            matUniforms->color = color;
            matUniforms->colorAmbient = Vector4(frandom(0.125), frandom(0.125), frandom(0.125), 0.0f);
            matUniforms->colorRim = Vector4(frandom(1.0), frandom(1.0), frandom(1.0), frandom(1.0));
            matUniforms->colorShadow = color;
            matUniforms->valuesRim = Vector4(0.125f, 0.0f, 1.0f, 0.25f);
            matUniforms->valuesSpecular = Vector4(0.0f, 1.0f, 1.0f, 0.0f);
        }

        // Material bind group (slot 1)
        m_materialBindingsLayout = rm->createBindGroupLayout({
            .textures = {
                {.slot = 1}, // Albedo
                {.slot = 2}, // Normal
                {.slot = 3}, // Properties
            },
            .buffers = {
                {.slot = 0, .stages = BindGroupLayoutDesc::STAGE_FLAGS::PIXEL}
            } });

        for (uint32 i = 0; i < NUM_MATERIALS; i++)
        {
            m_materialBindings[i] = rm->createBindGroup({
                .layout = m_materialBindingsLayout,
                .textures = {
                    // TODO: Create more textures and randomize
                    m_materialTextureAlbedo,
                    m_materialTextureNormal,
                    m_materialTextureProperties
                },
                .buffers = {{.buffer = m_materialUniforms[i]}} });
        }
    }

    void destroy(ResourceManager* rm)
    {
        auto timer = ScopedTimer("Materials destroy");
        for (uint32 i = 0; i < NUM_MATERIALS; i++) rm->deleteBindGroup(m_materialBindings[i]);
        rm->deleteBindGroupLayout(m_materialBindingsLayout);
        for (uint32 i = 0; i < NUM_MATERIALS; i++) rm->deleteBuffer(m_materialUniforms[i]);
        rm->deleteTexture(m_materialTextureProperties);
        rm->deleteTexture(m_materialTextureNormal);
        rm->deleteTexture(m_materialTextureAlbedo);
    }

    Handle<BindGroup> getMaterialBindGroup(uint32 index) const { return m_materialBindings[index]; }
    Handle<BindGroupLayout> getMaterialBindingsLayout() const { return m_materialBindingsLayout; }

private:
    Handle<Texture> m_materialTextureAlbedo;
    Handle<Texture> m_materialTextureNormal;
    Handle<Texture> m_materialTextureProperties;
    Handle<Buffer> m_materialUniforms[NUM_MATERIALS];
    Handle<BindGroupLayout> m_materialBindingsLayout;
    Handle<BindGroup> m_materialBindings[NUM_MATERIALS];
};

class ShadowRenderer
{
public:
    ShadowRenderer(ResourceManager* rm, TempRingUniformBuffer& uniformRingBuffer) : m_uniformRingBuffer(uniformRingBuffer)
        // TODO: Use GPU temp allocator instead of pre-allocated uniformRingBuffer. Requires BindGroup update.
    {
        auto timer = ScopedTimer("ShadowRenderer init");

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
        List<uint8> shadowShaderVS = ReadFile("../data/pack/gen/shader/hyper/mesh_shadow_vert.spv");

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

        // Draws
        // TODO: Implement culling
        CommandBuffer* commandBuffer = Renderer::ptr->beginCommandRecording(COMMAND_BUFFER_TYPE::OFFSCREEN);
        RenderPassRenderer* passRenderer = commandBuffer->beginRenderPass(m_renderPass, m_framebuffer);

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

class MainRenderer
{
public:
    MainRenderer(ResourceManager* rm, TempRingUniformBuffer& uniformRingBuffer, Handle<BindGroupLayout> materialBindingsLayout, Handle<Texture> sunShadowMap) :
        m_uniformRingBuffer(uniformRingBuffer)
        // TODO: Use GPU temp allocator instead of pre-allocated uniformRingBuffer. Requires BindGroup update.
    {
        auto timer = ScopedTimer("MainRenderer init");

        // Bindings layouts
        m_globalsBindingsLayout = rm->createBindGroupLayout({
            .textures = {
                {.slot = 1} // Global shadowmap
            },
            .buffers = {
                {.slot = 0} // Global uniforms (camera matrices, etc)
            } });

        m_drawBindingsLayout = ResourceManager::ptr->createBindGroupLayout(BindGroupLayoutDesc{
            .buffers = {
                {.slot = 0, .type = BindGroupLayoutDesc::BufferBinding::TYPE::UNIFORM_DYNAMIC_OFFSET, .stages = BindGroupLayoutDesc::STAGE_FLAGS::VERTEX} // Transform matrices
            } });

        // Shader
        List<uint8> shaderVS = ReadFile("../data/pack/gen/shader/hyper/mesh_simple_vert.spv");
        List<uint8> shaderPS = ReadFile("../data/pack/gen/shader/hyper/mesh_simple_frag.spv");

        m_shader = rm->createShader({
            .VS {.byteCode = shaderVS, .entryFunc = "main"},
            .PS {.byteCode = shaderPS, .entryFunc = "main"},
            .bindGroups = {
                { m_globalsBindingsLayout }, // Globals bind group (0)
                { materialBindingsLayout },  // Material bind group (1)
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
                    },
                    {
                        // Secondary vertex buffer: tangent, normal, color, texcoord (1)
                        .byteStride = 24, .attributes = {
                            {.byteOffset = 0,.format = FORMAT::RGBA16_FLOAT},
                            {.byteOffset = 8,.format = FORMAT::RGBA16_FLOAT},
                            {.byteOffset = 16,.format = FORMAT::RGBA8_UNORM},
                            {.byteOffset = 20,.format = FORMAT::RG16_FLOAT}
                        }
                    },
                },
                .renderPassLayout = rm->getBuiltinResources().mainRenderPassLayout
            } });

        // Uniform buffer
        m_globalUniforms = rm->createBuffer({ .byteSize = sizeof(GlobalUniforms) });
        m_uniforms = (GlobalUniforms*)rm->getBufferData(m_globalUniforms);
        m_uniforms->valuesFog = Vector4(/*fogEnd*/40.0f, /*decayPerMeter*/0.025f, 0.0f, 0.0f);
        m_uniforms->colorFog = Vector4(1.0f, 0.0f, 1.0f, 0.0f);

        // Bindings
        m_globalBindings = rm->createBindGroup({
            .layout = m_globalsBindingsLayout,
            .textures = {sunShadowMap},
            .buffers = {{.buffer = m_globalUniforms}} });

        m_drawBindings = rm->createBindGroup({
            .layout = m_drawBindingsLayout,
            .buffers = {{.buffer = uniformRingBuffer.getBuffer(), .byteSize = (uint32)sizeof(DrawUniforms)}} });
    }

    void destroy(ResourceManager* rm)
    {
        auto timer = ScopedTimer("MainRenderer destroy");
        rm->deleteBindGroup(m_drawBindings);
        rm->deleteBindGroup(m_globalBindings);
        rm->deleteBuffer(m_globalUniforms);
        rm->deleteShader(m_shader);
        rm->deleteBindGroupLayout(m_drawBindingsLayout);
        rm->deleteBindGroupLayout(m_globalsBindingsLayout);
    }

    void render(Span<const SceneObject> sceneObjects, const RenderFrame& frame, const SunLight& light, const Camera& camera)
    {
        // Update uniforms
        // TODO/FIXME: Needs to be double buffered!
        m_uniforms->view = camera.view;
        m_uniforms->viewProj = camera.viewProj;
        m_uniforms->viewInvTranspose = camera.viewInvTranspose;
        m_uniforms->lightDir = Vector4(light.direction, 0.0f);
        m_uniforms->lightViewProj = light.viewProj;

        // Draws
        // TODO: Implement culling
        CommandBuffer* commandBuffer = Renderer::ptr->beginCommandRecording(COMMAND_BUFFER_TYPE::MAIN);
        RenderPassRenderer* passRenderer = commandBuffer->beginRenderPass(frame.mainRenderPass, frame.framebuffer);

        List<Draw> draws((uint32)sceneObjects.size);
        for (const SceneObject& sceneObject : sceneObjects)
        {
            auto alloc = m_uniformRingBuffer.bumpAlloc<DrawUniforms>();
            Matrix3x4 mat = Matrix3x4::translate(sceneObject.position);
            alloc.ptr->model = mat;
            alloc.ptr->modelInvTranspose = Matrix3x3::inverse(Matrix3x3::from3x4(mat));

            draws.insert({
                .shader = m_shader, // TODO: Add more material shaders!
                .mesh = sceneObject.mesh,
                .bindGroup1 = sceneObject.material,
                .dynamicBufferOffset0 = alloc.offset });
        }

        DrawArea drawArea{ .bindGroup0 = m_globalBindings, .bindGroupDynamicOffsetBuffers = m_drawBindings, .drawOffset = 0, .drawCount = (uint32)sceneObjects.size };
        passRenderer->drawSubpass(drawArea, draws);

        commandBuffer->endRenderPass(passRenderer);
        commandBuffer->submit();

    }

private:
    TempRingUniformBuffer& m_uniformRingBuffer;
    GlobalUniforms* m_uniforms;

    Handle<BindGroupLayout> m_globalsBindingsLayout;
    Handle<BindGroupLayout> m_drawBindingsLayout;
    Handle<Shader> m_shader;
    Handle<Buffer> m_globalUniforms;
    Handle<BindGroup> m_globalBindings;
    Handle<BindGroup> m_drawBindings;
};

void TestDrawCallsSimple()
{
    ResourceManager* rm = ResourceManager::ptr;
    Device* device = Device::ptr;

    GPUAlignment alignment = device->getGPUAlignment();
    Vector2I surfaceDimensions = device->getSurfaceDimensions();
    float aspect = (float)surfaceDimensions.x / (float)surfaceDimensions.y;

    SunLight light;
    light.setDirection(Vector3(-0.57735f, 0.57735f, 0.57735f));

    Camera camera;
    camera.setPositionAspect(Vector3(-15.0f, 15.0f, 50.0f), aspect);

    auto meshes = Meshes(rm);
    auto materials = Materials(rm);

    // Scene: 10000 random objects
    auto timerInitScene = ScopedTimer("Init scene");
    const uint32 objCount = 10000;
    List<SceneObject> sceneObjects(objCount);
    for (uint32 i = 0; i < objCount; i++)
    {
        sceneObjects.insert({
            .position = Vector3(
                frandom(100.0f) - 50.0f,
                frandom(100.0f) - 50.0f,
                frandom(100.0f) - 50.0f),
            .material = materials.getMaterialBindGroup(random(Materials::NUM_MATERIALS)),
            .mesh = meshes.getMesh(random(Meshes::NUM_MESHES)) });
    }
    timerInitScene.end();

    // Uniform buffer containing all temp draw data
    // TODO/FIXME: Use temp allocator instead. Requires BindGroup update.
    auto uniformRingBuffer = TempRingUniformBuffer(rm, 1024 * 1024 * 16, alignment.uniformOffset);

    // Renderers
    auto shadowRenderer = ShadowRenderer(rm, uniformRingBuffer);
    auto mainRenderer = MainRenderer(rm, uniformRingBuffer, materials.getMaterialBindingsLayout(), shadowRenderer.getShadowMap());

    // Render 100 frames...
    auto timerRenderFrames = ScopedTimer("Render 100 frames");
    for (uint32 frameIndex = 0; frameIndex < 100; frameIndex++)
    {
        // Animate
        auto movementDirection = Vector3(0.0f, 0.0f, 1.0f);
        for (uint32 i = 0; i < objCount; i++)
        {
            sceneObjects[i].position += movementDirection * 0.1f;
        }

        float lightTime = (float)frameIndex * 0.001f + 123.71f;
        light.setDirection(Vector3(math::cos(lightTime), 0.57735f, -math::sin(lightTime)).getNormal());

        // Render
        RenderFrame frame = Renderer::ptr->beginFrame();
        shadowRenderer.render(sceneObjects, light);
        mainRenderer.render(sceneObjects, frame, light, camera);
        Renderer::ptr->present();
    }
    timerRenderFrames.end();

    // Cleanup
    auto timerCleanup = ScopedTimer("Cleanup");
    mainRenderer.destroy(rm);
    shadowRenderer.destroy(rm);
    uniformRingBuffer.destroy(rm);
    materials.destroy(rm);
    meshes.destroy(rm);
    timerCleanup.end();
}