package algorithms
import java.util.{Arrays => jArrays}

import gpu.utils.OpenCLInit
import org.jocl.CL._
import org.jocl._

import scala.Array.ofDim

object EulerTourDetection {

  private val memObjects = ofDim[cl_mem](3)
  private val eulerTourDetectionProgram =
    """__kernel void eulerTourDetect(__global const int *nodes,
      |                           __global const int *edges,
      |                           __global int *degree,
      |                           const int numberOfNodes,
      |                           const int numberOfEdges){
      |
      |    int gid = get_global_id(0);
      |    int startEdge = nodes[gid];
      |    int endEdge;
      |    if (gid + 1 < numberOfNodes){
      |       endEdge = nodes[gid + 1];
      |    } else {
      |       endEdge = numberOfEdges;
      |    }
      |
      |    int myDegree = (endEdge - startEdge) & 1;
      |    if (myDegree) {
      |       atomic_add(&degree[0],myDegree);
      |    }
      |
      |}""".stripMargin
  private var kernel : cl_kernel = _
  private var program : cl_program =_
  private var srcA, srcB, dst : Pointer = _
  private var nodeSize, edgeSize : Int = 0
  private var dstArray = ofDim[Int](3)

  def EulerTourInit(srcArrayNodes : Array[Int], srcArrayEdges: Array[Int]): Unit = {

    nodeSize = srcArrayNodes.length
    edgeSize = srcArrayEdges.length

    srcA = Pointer.to(srcArrayNodes)
    srcB = Pointer.to(srcArrayEdges)
    dst = Pointer.to(dstArray)

    memObjects(0) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * nodeSize, srcA, null)
    memObjects(1) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * edgeSize, srcB, null)
    memObjects(2) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_WRITE_ONLY,
      Sizeof.cl_int * nodeSize, null, null)

    // Create the program from the source code
    program = clCreateProgramWithSource(OpenCLInit.getContext, 1, Array(eulerTourDetectionProgram),
      null, null)

    // Build the program
    clBuildProgram(program, 0, null, null, null, null)

    // Create the kernel
    kernel = clCreateKernel(program, "eulerTourDetect", null)
  }

  def eulerTourDetection(): Boolean ={

    // Set the arguments for the kernel
    for (i <- memObjects.indices) {
      clSetKernelArg(kernel, i, Sizeof.cl_mem, Pointer.to(memObjects(i)))
    }
    clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(Array[Int](nodeSize)))
    clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(Array[Int](edgeSize)))


    // Set the work-item dimensions
    val global_work_size = Array(nodeSize.toLong)
    val local_work_size = Array(1L)

    // Execute the kernel
    clEnqueueNDRangeKernel(OpenCLInit.getCommandQueue, kernel, 1, null, global_work_size,
      local_work_size, 0, null, null)

    // Read the output data
    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(2), CL_TRUE, 0,
      Sizeof.cl_int * nodeSize, dst , 0, null, null)

    // Hay camino euleriano en caso de tener 2 o menos nodos con grado impar
    dstArray(0) == 2 || dstArray(0) == 0
  }

  def destroy(): Unit ={
    clReleaseKernel(kernel)
    clReleaseProgram(program)
  }

}
