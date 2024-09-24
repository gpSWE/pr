const wgsl = `

@group( 0 ) @binding( 0 ) var<storage, read> input: array<f32>;
@group( 0 ) @binding( 1 ) var<storage, read_write> output: array<f32>;
var<workgroup> sharedMemory: array<f32, 256>;

@compute @workgroup_size( 256 ) fn sum(
	@builtin( global_invocation_id ) global_id: vec3<u32>,
	@builtin( local_invocation_id ) local_id: vec3<u32>,
	@builtin( workgroup_id ) group_id: vec3<u32>
) {

	let globalIndex = global_id.x;
	let localIndex = local_id.x;
	let groupSize = 256u;
	
	if ( globalIndex < arrayLength( &input ) ) {
		sharedMemory[ localIndex ] = input[ globalIndex ];
	}
	else {
		sharedMemory[ localIndex ] = 0.0;
	}
	
	workgroupBarrier();
	
	for ( var stride = groupSize / 2u; stride > 0u; stride /= 2u ) {
		if ( localIndex < stride ) {
			sharedMemory[ localIndex ] += sharedMemory[ localIndex + stride ];
		}
		workgroupBarrier();
	}

	if ( localIndex == 0u ) {
		output[ group_id.x ] = sharedMemory[ 0 ];
	}
}
`

async function computeSum( inputData ) {

	const adapter = await navigator.gpu.requestAdapter()
	const device = await adapter.requestDevice()
	const module = device.createShaderModule( { code: wgsl } )

	const length = inputData.length

	const inputBuffer = device.createBuffer( {
		size: inputData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		mappedAtCreation: true,
	} )

	new Float32Array( inputBuffer.getMappedRange() ).set( inputData )
	inputBuffer.unmap()

	const workgroupCount = Math.ceil( length / 256 )
	const outputBuffer = device.createBuffer( {
		size: Float32Array.BYTES_PER_ELEMENT * workgroupCount,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	} )

	const stagingBuffer = device.createBuffer( {
		size: Float32Array.BYTES_PER_ELEMENT * workgroupCount,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	} )

	const bindGroupLayout = device.createBindGroupLayout( {
		entries: [
			{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
			{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
		],
	} )

	const bindGroup = device.createBindGroup( {
		layout: bindGroupLayout,
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	} )

	const computePipeline = device.createComputePipeline( {
		layout: device.createPipelineLayout( { bindGroupLayouts: [ bindGroupLayout ] } ),
		compute: { module, entryPoint: "sum" },
	} )

	const commandEncoder = device.createCommandEncoder()
	const passEncoder = commandEncoder.beginComputePass()
	passEncoder.setPipeline( computePipeline )
	passEncoder.setBindGroup( 0, bindGroup )
	passEncoder.dispatchWorkgroups( workgroupCount )
	passEncoder.end()

	commandEncoder.copyBufferToBuffer( outputBuffer, 0, stagingBuffer, 0, Float32Array.BYTES_PER_ELEMENT * workgroupCount )

	device.queue.submit( [ commandEncoder.finish() ] )

	await stagingBuffer.mapAsync( GPUMapMode.READ )
	const resultArrayBuffer = stagingBuffer.getMappedRange()
	const result = new Float32Array( resultArrayBuffer )

	// Sum up the partial results from each workgroup
	let finalSum = 0
	for ( let i = 0; i < result.length; i++ ) {
		finalSum += result[ i ]
	}

	stagingBuffer.unmap()

	return finalSum
}

const inputData = new Float32Array( Array.from( { length: 1_048_576 }, ( _, i ) => i + 1 ) )

const startTime = performance.now()

computeSum( inputData ).then( sum => {

	const endTime = performance.now()

	console.log( "SUM:", sum )
	console.log( "Milliseconds:", ( endTime - startTime ).toFixed( 2 )  )
} )
