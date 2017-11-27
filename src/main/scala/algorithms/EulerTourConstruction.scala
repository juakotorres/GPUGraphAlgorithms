package algorithms

import java.util.{Arrays => jArrays}
import gpu.utils.OpenCLInit
import org.jocl.CL._
import org.jocl._

import scala.Array.ofDim

object EulerTourConstruction {

  private val memObjects = ofDim[cl_mem](10)

  private val eulerTourSorting =
    """
      |__kernel void eulerTourSorting(__global const int *vertex1,
      |                            __global const int *vertex2,
      |                            __global int *v1v2){
      |
      |  int gid = get_global_id(0);
      |  v1v2[2*(gid - 1) + 1] = vertex1[gid];
      |  v1v2[2*gid] = vertex2[gid];
      |
      |}
    """.stripMargin

  private val eulerTourBoundaries =
    """
      |__kernel void eulerTourBoundaries(__global int *v1v2,
      |                                  __global int *boundaries,
      |                                  const int m){
      |
      |    int gid = get_global_id(0);
      |
      |    if (gid < m - 1) {
      |       if(v1v2[2*gid] != v1v2[2*gid + 1]) {
      |         boundaries[v1v2[2*gid]] = 2*gid + 1;
      |       } else if(v1v2[2*gid + 1] != v1v2[2*gid + 2]){
      |         boundaries[v1v2[2*gid + 1]] = 2*gid + 2;
      |       }
      |    } else {
      |       boundaries[v1v2[2*gid]] = 2*m;
      |    }
      |
      |}
    """.stripMargin

  private val eulerTourGetDegree =
    """
      |__kernel void eulerTourGetDegree(__global int *boundaries,
      |                                  __global int *degree,
      |                                  const int n){
      |
      |    int gid = get_global_id(0);
      |
      |    if(gid == 0){
      |       degree[gid] = boundaries[gid];
      |    }
      |
      |    if (gid != 0 && gid < n) {
      |       degree[gid] = boundaries[gid] - boundaries[gid - 1];
      |    }
      |
      |    degree[gid] = degree[gid]/2 + degree[gid] % 2;
      |}
    """.stripMargin

  private val prefixSum =
    """
      |__kernel void prefixSum(__global int *degree,
      |                        __global int *S,
      |                        const int n){
      |
      |    int gid = get_global_id(0);
      |
      |    if (gid < n) {
      |       S[gid] = degree[gid];
      |       for(int i = 0; i < gid; i++){
      |         S[gid] = S[gid] + degree[i];
      |       }
      |    }
      |
      |}
    """.stripMargin

  private val fillTable =
    """
      |__kernel void fillTable(__global int *N,
      |                        const int size){
      |
      |    int gid = get_global_id(0);
      |
      |    if (N[gid] == 0) {
      |       for(int i = gid + 1; i < size; i++) {
      |         if(N[i] != 0) {
      |           N[gid] = N[i];
      |           break;
      |         }
      |       }
      |    }
      |
      |}
    """.stripMargin

  private val correspondenceTable =
    """
      |__kernel void N(__global int *S,
      |                __global int *N,
      |                const int n,
      |                const int size){
      |
      |    int gid = get_global_id(0);
      |    if (gid < size) {
      |       N[gid] = 0;
      |    }
      |
      |    if (gid < n) {
      |       int u = S[gid];
      |       N[u - 1] = gid + 1;
      |    }
      |
      |    N[0] = 1;
      |
      |}
    """.stripMargin

  private val eulerTourDetectionProgram =
    """__kernel void eulerTourDetect(__global const int *vertex1,
      |                           __global const int *vertex2,
      |                           __global const int *boundaries,
      |                           __global int *degree,
      |                           __global const int *v1v2,
      |                           __global int *N,
      |                           __global int *S,
      |                           __global int *nv1v2,
      |                           __global int *nv1,
      |                           __global int *nv2,
      |                           const int n,
      |                           const int m){
      |
      |    int gid = get_global_id(0);
      |
      |    if(gid == 0){
      |       degree[gid] = boundaries[gid];
      |    }
      |
      |    if (gid != 0 && gid < n) {
      |       degree[gid] = boundaries[gid] - boundaries[gid - 1];
      |    }
      |
      |    nv1[gid] = gid;
      |
      |
      |}""".stripMargin
  private var kernel, kernel2, kernel3,
              kernel4, kernel5, kernel6,
              kernel7 : cl_kernel = _
  private var program, program2, program3,
              program4, program5, program6,
              program7 : cl_program =_
  private var srcA, srcB, v1, v2 : Pointer = _
  private var n, m : Int = 0
  private var v1Array, v2Array : Array[Int] = _
  private var NSize : Int = _

  def EulerTourInit(vertex1Array : Array[Int], vertex2Array: Array[Int], numberOfNodes : Int): Unit = {

    n = numberOfNodes
    println(n)
    m = vertex2Array.length

    v1Array = ofDim[Int](m)
    v2Array = ofDim[Int](m)

    srcA = Pointer.to(vertex1Array)
    srcB = Pointer.to(vertex2Array)
    v1 = Pointer.to(v1Array)
    v2 = Pointer.to(v2Array)

    memObjects(0) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * m, srcA, null)
    memObjects(1) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * m, srcB, null)
    // Boundaries
    memObjects(2) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_WRITE_ONLY,
      Sizeof.cl_int * n, null, null)
    // Degree
    memObjects(3) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_WRITE_ONLY,
      Sizeof.cl_int * n, null, null)
    // v1v2
    memObjects(4) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_WRITE_ONLY,
      Sizeof.cl_int * 2 * m, null, null)
    // S
    memObjects(6) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_WRITE_ONLY,
      Sizeof.cl_int * n, null, null)
    // nv1v2
    memObjects(7) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_WRITE_ONLY,
      Sizeof.cl_int * 2 * m, null, null)
    // nv1
    memObjects(8) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_WRITE_ONLY,
      Sizeof.cl_int * m, null, null)
    // nv2
    memObjects(9) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_WRITE_ONLY,
      Sizeof.cl_int * m, null, null)

    // Create the program from the source code
    program = clCreateProgramWithSource(OpenCLInit.getContext, 1, Array(eulerTourDetectionProgram),
      null, null)
    // Create the program from the source code
    program2 = clCreateProgramWithSource(OpenCLInit.getContext, 1, Array(eulerTourSorting),
      null, null)
    // Create the program from the source code
    program3 = clCreateProgramWithSource(OpenCLInit.getContext, 1, Array(eulerTourBoundaries),
      null, null)
    // Create the program from the source code
    program4 = clCreateProgramWithSource(OpenCLInit.getContext, 1, Array(eulerTourGetDegree),
      null, null)
    // Create the program from the source code
    program5 = clCreateProgramWithSource(OpenCLInit.getContext, 1, Array(prefixSum),
      null, null)
    // Create the program from the source code
    program6 = clCreateProgramWithSource(OpenCLInit.getContext, 1, Array(correspondenceTable),
      null, null)
    // Create the program from the source code
    program7 = clCreateProgramWithSource(OpenCLInit.getContext, 1, Array(fillTable),
      null, null)



    // Build the program
    clBuildProgram(program, 0, null, null, null, null)
    clBuildProgram(program2, 0, null, null, null, null)
    clBuildProgram(program3, 0, null, null, null, null)
    clBuildProgram(program4, 0, null, null, null, null)
    clBuildProgram(program5, 0, null, null, null, null)
    clBuildProgram(program6, 0, null, null, null, null)
    clBuildProgram(program7, 0, null, null, null, null)

    // Create the kernel
    kernel = clCreateKernel(program, "eulerTourDetect", null)
    kernel2 = clCreateKernel(program2, "eulerTourSorting", null)
    kernel3 = clCreateKernel(program3, "eulerTourBoundaries", null)
    kernel4 = clCreateKernel(program4, "eulerTourGetDegree", null)
    kernel5 = clCreateKernel(program5, "prefixSum", null)
    kernel6 = clCreateKernel(program6, "N", null)
    kernel7 = clCreateKernel(program7, "fillTable", null)
  }

  def eulerTourCalculateBoundaries(): Unit = {

    // Set the work-item dimensions
    val global_work_size = Array(m.toLong)
    val local_work_size = Array(1L)

    clSetKernelArg(kernel3, 0, Sizeof.cl_mem, Pointer.to(memObjects(4)))
    clSetKernelArg(kernel3, 1, Sizeof.cl_mem, Pointer.to(memObjects(2)))
    clSetKernelArg(kernel3, 2, Sizeof.cl_int, Pointer.to(Array[Int](m)))

    // Execute the kernel
    clEnqueueNDRangeKernel(OpenCLInit.getCommandQueue, kernel3, 1, null, global_work_size,
      local_work_size, 0, null, null)

    val boundaries = ofDim[Int](n)

    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(2), CL_TRUE, 0,
      Sizeof.cl_int * n, Pointer.to(boundaries) , 0, null, null)

    // v1v2
    memObjects(2) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * n, Pointer.to(boundaries), null)

    println(jArrays.toString(boundaries))
  }


  def eulerTourConstructV1V2(): Unit = {

    // Set the work-item dimensions
    val global_work_size = Array(m.toLong)
    val local_work_size = Array(1L)

    clSetKernelArg(kernel2, 0, Sizeof.cl_mem, Pointer.to(memObjects(0)))
    clSetKernelArg(kernel2, 1, Sizeof.cl_mem, Pointer.to(memObjects(1)))
    clSetKernelArg(kernel2, 2, Sizeof.cl_mem, Pointer.to(memObjects(4)))

    // Execute the kernel
    clEnqueueNDRangeKernel(OpenCLInit.getCommandQueue, kernel2, 1, null, global_work_size,
      local_work_size, 0, null, null)

    val v1v2Array = ofDim[Int](2*m)

    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(4), CL_TRUE, 0,
      Sizeof.cl_int * 2 * m, Pointer.to(v1v2Array) , 0, null, null)

    // v1v2
    memObjects(4) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * 2 * m, Pointer.to(v1v2Array.sorted), null)

    println(jArrays.toString(v1v2Array.sorted))
  }

  def eulerTourDegree() : Unit = {
    // Set the work-item dimensions
    val global_work_size = Array(m.toLong)
    val local_work_size = Array(1L)

    clSetKernelArg(kernel4, 0, Sizeof.cl_mem, Pointer.to(memObjects(2)))
    clSetKernelArg(kernel4, 1, Sizeof.cl_mem, Pointer.to(memObjects(3)))
    clSetKernelArg(kernel4, 2, Sizeof.cl_int, Pointer.to(Array[Int](n)))

    // Execute the kernel
    clEnqueueNDRangeKernel(OpenCLInit.getCommandQueue, kernel4, 1, null, global_work_size,
      local_work_size, 0, null, null)

    val degree = ofDim[Int](n)

    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(3), CL_TRUE, 0,
      Sizeof.cl_int * n, Pointer.to(degree) , 0, null, null)

    // v1v2
    memObjects(3) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * n, Pointer.to(degree), null)

    println(jArrays.toString(degree))
  }

  def prefixSumDegree() : Unit = {
    // Set the work-item dimensions
    val global_work_size = Array(m.toLong)
    val local_work_size = Array(1L)

    clSetKernelArg(kernel5, 0, Sizeof.cl_mem, Pointer.to(memObjects(3)))
    clSetKernelArg(kernel5, 1, Sizeof.cl_mem, Pointer.to(memObjects(6)))
    clSetKernelArg(kernel5, 2, Sizeof.cl_int, Pointer.to(Array[Int](n)))

    // Execute the kernel
    clEnqueueNDRangeKernel(OpenCLInit.getCommandQueue, kernel5, 1, null, global_work_size,
      local_work_size, 0, null, null)

    val prefixSum = ofDim[Int](n)

    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(6), CL_TRUE, 0,
      Sizeof.cl_int * n, Pointer.to(prefixSum) , 0, null, null)

    // v1v2
    memObjects(6) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * n, Pointer.to(prefixSum), null)


    //Creamos el buffer para N
    NSize = prefixSum(n - 1)
    memObjects(5) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_WRITE_ONLY,
      Sizeof.cl_int * NSize, null, null)

    println(jArrays.toString(prefixSum))
  }

  def getCorrespondenceTable() : Unit = {
    // Set the work-item dimensions
    val global_work_size = Array(m.toLong)
    val local_work_size = Array(1L)

    clSetKernelArg(kernel6, 0, Sizeof.cl_mem, Pointer.to(memObjects(6)))
    clSetKernelArg(kernel6, 1, Sizeof.cl_mem, Pointer.to(memObjects(5)))
    clSetKernelArg(kernel6, 2, Sizeof.cl_int, Pointer.to(Array[Int](n)))
    clSetKernelArg(kernel6, 3, Sizeof.cl_int, Pointer.to(Array[Int](NSize)))

    // Execute the kernel
    clEnqueueNDRangeKernel(OpenCLInit.getCommandQueue, kernel6, 1, null, global_work_size,
      local_work_size, 0, null, null)

    val N = ofDim[Int](NSize)

    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(5), CL_TRUE, 0,
      Sizeof.cl_int * NSize, Pointer.to(N) , 0, null, null)

    // N
    memObjects(5) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * NSize, Pointer.to(N), null)

    println(jArrays.toString(N))
  }

  def fillTableN() : Unit = {
    // Set the work-item dimensions
    val global_work_size = Array(m.toLong)
    val local_work_size = Array(1L)

    clSetKernelArg(kernel7, 0, Sizeof.cl_mem, Pointer.to(memObjects(5)))
    clSetKernelArg(kernel7, 1, Sizeof.cl_int, Pointer.to(Array[Int](NSize)))

    // Execute the kernel
    clEnqueueNDRangeKernel(OpenCLInit.getCommandQueue, kernel7, 1, null, global_work_size,
      local_work_size, 0, null, null)

    val N = ofDim[Int](NSize)

    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(5), CL_TRUE, 0,
      Sizeof.cl_int * NSize, Pointer.to(N) , 0, null, null)

    // N
    memObjects(5) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * NSize, Pointer.to(N), null)

    println(jArrays.toString(N))
  }

  def eulerTour(): Unit ={

    // Set the work-item dimensions
    val global_work_size = Array(m.toLong)
    val local_work_size = Array(1L)

    eulerTourConstructV1V2()
    eulerTourCalculateBoundaries()
    eulerTourDegree()
    prefixSumDegree()
    getCorrespondenceTable()
    fillTableN()

    // Set the arguments for the kernel
    for (i <- memObjects.indices) {
      clSetKernelArg(kernel, i, Sizeof.cl_mem, Pointer.to(memObjects(i)))
    }
    clSetKernelArg(kernel, 10, Sizeof.cl_int, Pointer.to(Array[Int](n)))
    clSetKernelArg(kernel, 11, Sizeof.cl_int, Pointer.to(Array[Int](m)))

    // Execute the kernel
    clEnqueueNDRangeKernel(OpenCLInit.getCommandQueue, kernel, 1, null, global_work_size,
      local_work_size, 0, null, null)

    // Read the output data
    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(8), CL_TRUE, 0,
      Sizeof.cl_int * m, v1 , 0, null, null)
    /*clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(9), CL_TRUE, 0,
      Sizeof.cl_int * m, v2 , 0, null, null)*/

    val degree = ofDim[Int](n)


    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(3), CL_TRUE, 0,
      Sizeof.cl_int * n, Pointer.to(degree) , 0, null, null)
  }

  def destroy(): Unit ={
    clReleaseKernel(kernel)
    clReleaseProgram(program)
  }
}
