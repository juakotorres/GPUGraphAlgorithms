import java.util.{Arrays => jArrays}

import org.jocl.CL._
import org.jocl._

import scala.Array.ofDim

/**
  * A small JOCL sample.
  */
object JOCLSample {
  /**
    * The source code of the OpenCL program to execute
    */
  private val programSource =
    """__kernel void sampleKernel(__global const float *a,
      |                           __global const float *b,
      |                           __global float *c){
      |    int gid = get_global_id(0);
      |    c[gid] = a[gid] * b[gid];
      |}""".stripMargin

  /**
    * The entry point of this sample
    *
    * @param args Not used
    */
  def run(args: Array[String]): Unit = {
    // Create input- and output data
    val n = 10
    val srcArrayA, srcArrayB, dstArray = ofDim[Float](n)
    for (i <- 0 until n) {
      srcArrayA(i) = i
      srcArrayB(i) = i
    }
    val srcA = Pointer.to(srcArrayA)
    val srcB = Pointer.to(srcArrayB)
    val dst = Pointer.to(dstArray)

    // The platform, device type and device number
    // that will be used
    val platformIndex = 0
    val deviceType = CL_DEVICE_TYPE_ALL
    val deviceIndex = 0

    // Enable exceptions and subsequently omit error checks in this sample
    setExceptionsEnabled(true)

    val platform = {
      // Obtain the number of platforms
      val numPlatformsArray = Array(0)
      clGetPlatformIDs(0, null, numPlatformsArray)
      val numPlatforms = numPlatformsArray(0)

      // Obtain a platform ID
      val platforms = ofDim[cl_platform_id](numPlatforms)
      clGetPlatformIDs(platforms.length, platforms, null)
      platforms(platformIndex)
    }

    // Initialize the context properties
    val contextProperties = new cl_context_properties()
    contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform)

    val device = {
      // Obtain the number of devices for the platform
      val numDevicesArray = Array(0)
      clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
      val numDevices = numDevicesArray(0)

      // Obtain a device ID
      val devices = ofDim[cl_device_id](numDevices)
      clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
      devices(deviceIndex)
    }

    // Create a context for the selected device
    val context = clCreateContext(contextProperties, 1, Array(device),
      null, null, null)

    // Create a command-queue for the selected device
    val commandQueue = clCreateCommandQueue(context, device, 0, null)

    // Allocate the memory objects for the input- and output data
    val memObjects = ofDim[cl_mem](3)
    memObjects(0) = clCreateBuffer(context,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_float * n, srcA, null)
    memObjects(1) = clCreateBuffer(context,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_float * n, srcB, null)
    memObjects(2) = clCreateBuffer(context,
      CL_MEM_READ_WRITE,
      Sizeof.cl_float * n, null, null)

    // Create the program from the source code
    val program = clCreateProgramWithSource(context, 1, Array(programSource),
      null, null)

    // Build the program
    clBuildProgram(program, 0, null, null, null, null)

    // Create the kernel
    val kernel = clCreateKernel(program, "sampleKernel", null)

    // Set the arguments for the kernel
    for (i <- memObjects.indices) {
      clSetKernelArg(kernel, i, Sizeof.cl_mem, Pointer.to(memObjects(i)))
    }

    // Set the work-item dimensions
    val global_work_size = Array(n.toLong)
    val local_work_size = Array(1L)

    // Execute the kernel
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size,
      local_work_size, 0, null, null)

    // Read the output data
    clEnqueueReadBuffer(commandQueue, memObjects(2), CL_TRUE, 0,
      n * Sizeof.cl_float, dst, 0, null, null)

    // Release kernel, program, and memory objects
    memObjects.foreach(clReleaseMemObject)
    clReleaseKernel(kernel)
    clReleaseProgram(program)
    clReleaseCommandQueue(commandQueue)
    clReleaseContext(context)

    // Verify the result
    val epsilon = 1e-7f
    val passed = dstArray.zip(srcArrayA.zip(srcArrayB).map(p => p._1 * p._2))
      .forall{ case (x, y) => Math.abs(x - y) <= epsilon * Math.abs(x) }
    println(s"Test ${if (passed) "PASSED" else "FAILED"}")
    if (n <= 10) {
      println(s"Result: ${jArrays.toString(dstArray)}")
    }
  }
}
