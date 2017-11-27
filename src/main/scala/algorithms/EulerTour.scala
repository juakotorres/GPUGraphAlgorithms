package algorithms
import java.util.{Arrays => jArrays}

import gpu.utils.OpenCLInit
import org.jocl.CL._
import org.jocl._

import scala.Array.ofDim

object EulerTour {

  private val memObjects = ofDim[cl_mem](3)
  private val programSource =
    """__kernel void eulerTourDetect(__global const int *nodes,
      |                           __global const int *edges,
      |                           __global const int numberOfNodes,
      |                           __global const int numberOfEdges,
      |                           __global int oddDegree){
      |    int gid = get_global_id(0);
      |    int startEdgeIndex = nodes[gid];
      |    int endEdgeIndex = gid + 1 == numberOfNodes? numberOfEdges : edges[gid];
      |
      |    int degree = ((endEdgeIndex - startEdgeIndex) & 1) == 0? 0 : 1;
      |
      |    if (degree) {
      |       atomic_add(&oddDegree, 1)
      |    }
      |
      |}""".stripMargin
  private var kernel : cl_kernel = _
  private var program : cl_program =_
  private var graphSize : Int = 0
  var srcA, srcB : Pointer = _
  var dst, nodeSize, edgeSize : Int = 0

  def EulerTourInit(srcArrayNodes : Array[Float], srcArrayEdges: Array[Float]): Unit = {

    nodeSize = srcArrayNodes.length
    edgeSize = srcArrayEdges.length

    srcA = Pointer.to(srcArrayNodes)
    srcB = Pointer.to(srcArrayEdges)

    memObjects(0) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_float * srcArrayNodes.length, srcA, null)
    memObjects(1) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_float * srcArrayEdges.length, srcB, null)

    // Create the program from the source code
    program = clCreateProgramWithSource(OpenCLInit.getContext, 1, Array(programSource),
      null, null)

    // Build the program
    clBuildProgram(program, 0, null, null, null, null)

    // Create the kernel
    kernel = clCreateKernel(program, "eulerTourDetect", null)
  }

  def eulerTourDetection(): Unit ={

    // Set the arguments for the kernel
    for (i <- memObjects.indices) {
      clSetKernelArg(kernel, i, Sizeof.cl_mem, Pointer.to(memObjects(i)))
    }
    clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(Array[Int](dst)))
    clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(Array[Int](dst)))
    clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(Array[Int](dst)))

    // Set the work-item dimensions
    val global_work_size = Array(graphSize.toLong)
    val local_work_size = Array(1L)

    // Execute the kernel
    clEnqueueNDRangeKernel(OpenCLInit.getCommandQueue, kernel, 1, null, global_work_size,
      local_work_size, 0, null, null)

    // Read the output data
    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(2), CL_TRUE, 0,
      Sizeof.cl_int, Pointer.to(Array[Int](dst)) , 0, null, null)

    println(dst)
  }

  def destroy(): Unit ={
    clReleaseKernel(kernel)
    clReleaseProgram(program)
  }

}
