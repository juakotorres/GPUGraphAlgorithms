package algorithms

import java.util.{Arrays => jArrays}

import gpu.utils.OpenCLInit
import org.jocl.CL._
import org.jocl._
import util.control.Breaks._

import scala.Array.ofDim
import scala.collection.mutable

object EulerTourConstruction {

  private val memObjects = ofDim[cl_mem](10)
  private val memDegree = ofDim[cl_mem](2)

  private val eulerTourSorting =
    """
      |__kernel void eulerTourSorting(__global const int *vertex1,
      |                            __global const int *vertex2,
      |                            __global int *v1v2){
      |
      |  int gid = get_global_id(0);
      |  v1v2[2*gid] = vertex1[gid];
      |  v1v2[2*gid + 1] = vertex2[gid];
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
      |         boundaries[v1v2[2*gid] - 1] = 2*gid + 1;
      |       } else if(v1v2[2*gid + 1] != v1v2[2*gid + 2]){
      |         boundaries[v1v2[2*gid + 1] - 1] = 2*gid + 2;
      |       }
      |    } else {
      |       boundaries[v1v2[2*gid] - 1] = 2*m;
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
      |}
    """.stripMargin

  private val getOddAndDegree =
    """
      |__kernel void getOddAndDegree(__global int *degree,
      |                                 __global int *ndegree,
      |                                 __global int *odd,
      |                                  const int n){
      |
      |    int gid = get_global_id(0);
      |
      |    if(gid > 0 && gid < n && degree[gid] % 2 == 1 && degree[gid - 1] % 2 != 1){
      |       odd[0] = gid + 1;
      |    }
      |
      |    ndegree[gid] = degree[gid]/2 + degree[gid] % 2;
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
    """__kernel void eulerTourDetect(__global const int *boundaries,
      |                           __global const int *v1v2,
      |                           __global int *S,
      |                           __global int *nv1v2,
      |                           const int odd,
      |                           const int m){
      |
      |    int gid = get_global_id(0);
      |
      |    int e1 = 2*gid;
      |    int e2 = 2*gid + 1;
      |    int sum1 = e1 + 1;
      |    int sum2 = e2 + 1;
      |
      |    if(e1 == odd){
      |       sum1 = e1 + 2;
      |    } else if (e2 == odd) {
      |       sum2 = e2 + 2;
      |    }
      |
      |    int v1v2e1 = v1v2[e1] - 1;
      |    int v1v2e2 = v1v2[e2] - 1;
      |    int se1 = 0;
      |    int se2 = 0;
      |    int be1 = 0;
      |    int be2 = 0;
      |
      |    if (v1v2e1 >= 0){
      |       se1 = S[v1v2e1];
      |       be1 = boundaries[v1v2e1];
      |    }
      |
      |    if (v1v2e2 >= 0){
      |       se2 = S[v1v2e2];
      |       be2 = boundaries[v1v2e2];
      |    }
      |
      |    nv1v2[e1] = se1 + (sum1 - be1)/2;
      |    nv1v2[e2] = se2 + (sum2 - be2)/2;
      |
      |
      |}""".stripMargin
  private var kernel, kernel2, kernel3,
              kernel4, kernel5, kernel6,
              kernel7, kernel8 : cl_kernel = _
  private var program, program2, program3,
              program4, program5, program6,
              program7, program8 : cl_program =_
  private var srcA, srcB, v1, v2 : Pointer = _
  private var n, m : Int = 0
  private var v1Array, v2Array : Array[Int] = _
  private var NSize : Int = _
  private var oddNode : Int = _
  private var NArray : Array[Int] = _
  private var V1x, V2x : Array[Int] = _

  def EulerTourInit(vertex1Array : Array[Int], vertex2Array: Array[Int], numberOfNodes : Int): Unit = {

    n = numberOfNodes
    println("number of nodes: "+ n)
    println("Array v1: " + jArrays.toString(vertex1Array))
    println("Array v2: " + jArrays.toString(vertex2Array))
    V1x = vertex1Array
    V2x = vertex2Array
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
    // Create the program from the source code
    program8 = clCreateProgramWithSource(OpenCLInit.getContext, 1, Array(getOddAndDegree),
      null, null)



    // Build the program
    clBuildProgram(program, 0, null, null, null, null)
    clBuildProgram(program2, 0, null, null, null, null)
    clBuildProgram(program3, 0, null, null, null, null)
    clBuildProgram(program4, 0, null, null, null, null)
    clBuildProgram(program5, 0, null, null, null, null)
    clBuildProgram(program6, 0, null, null, null, null)
    clBuildProgram(program7, 0, null, null, null, null)
    clBuildProgram(program8, 0, null, null, null, null)

    // Create the kernel
    kernel = clCreateKernel(program, "eulerTourDetect", null)
    kernel2 = clCreateKernel(program2, "eulerTourSorting", null)
    kernel3 = clCreateKernel(program3, "eulerTourBoundaries", null)
    kernel4 = clCreateKernel(program4, "eulerTourGetDegree", null)
    kernel5 = clCreateKernel(program5, "prefixSum", null)
    kernel6 = clCreateKernel(program6, "N", null)
    kernel7 = clCreateKernel(program7, "fillTable", null)
    kernel8 = clCreateKernel(program8, "getOddAndDegree", null)
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

    println("boundaries: " + jArrays.toString(boundaries))
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

    println("v1v2 array: " + jArrays.toString(v1v2Array.sorted))
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

    // degree
    memObjects(3) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * n, Pointer.to(degree), null)

    println("Original Degree:" + jArrays.toString(degree))
  }


  def getOddAndDegreeCall() : Unit = {
    // Set the work-item dimensions
    val global_work_size = Array(n.toLong)
    val local_work_size = Array(1L)

    val nDegree = ofDim[Int](n)
    val odd = ofDim[Int](n)

    memDegree(0) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * n, Pointer.to(nDegree), null)

    memDegree(1) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * n, Pointer.to(odd), null)


    clSetKernelArg(kernel8, 0, Sizeof.cl_mem, Pointer.to(memObjects(3)))
    clSetKernelArg(kernel8, 1, Sizeof.cl_mem, Pointer.to(memDegree(0)))
    clSetKernelArg(kernel8, 2, Sizeof.cl_mem, Pointer.to(memDegree(1)))
    clSetKernelArg(kernel8, 3, Sizeof.cl_int, Pointer.to(Array[Int](n)))

    // Execute the kernel
    clEnqueueNDRangeKernel(OpenCLInit.getCommandQueue, kernel8, 1, null, global_work_size,
      local_work_size, 0, null, null)

    val degree = ofDim[Int](n)

    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memDegree(0), CL_TRUE, 0,
      Sizeof.cl_int * n, Pointer.to(degree) , 0, null, null)
    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memDegree(1), CL_TRUE, 0,
      Sizeof.cl_int * n, Pointer.to(odd) , 0, null, null)

    // v1v2
    memObjects(3) = clCreateBuffer(OpenCLInit.getContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      Sizeof.cl_int * n, Pointer.to(degree), null)

    oddNode = odd(0)
    println("Degree : " + jArrays.toString(degree))
    println("Odd vertex : " + odd(0))
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

    println("S Array (PrefixSum): "+ jArrays.toString(prefixSum))
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

    println("Correspondence table: " + jArrays.toString(N))
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

    NArray = N

    println("Ready fill table: " + jArrays.toString(N))
  }

  def eulerTour(): Unit ={

    // Set the work-item dimensions
    val global_work_size = Array(m.toLong)
    val local_work_size = Array(1L)

    eulerTourConstructV1V2()
    eulerTourCalculateBoundaries()
    eulerTourDegree()
    getOddAndDegreeCall()
    prefixSumDegree()
    getCorrespondenceTable()
    fillTableN()

    clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memObjects(2)))
    clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memObjects(4)))
    clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memObjects(6)))
    clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(memObjects(7)))
    clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(Array[Int](oddNode)))
    clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(Array[Int](m)))

    // Execute the kernel
    clEnqueueNDRangeKernel(OpenCLInit.getCommandQueue, kernel, 1, null, global_work_size,
      local_work_size, 0, null, null)

    // Read the output data
    /*clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(8), CL_TRUE, 0,
      Sizeof.cl_int * m, v1 , 0, null, null)
    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(9), CL_TRUE, 0,
      Sizeof.cl_int * m, v2 , 0, null, null)*/

    val nv1v2 = ofDim[Int](2*m)


    clEnqueueReadBuffer(OpenCLInit.getCommandQueue, memObjects(7), CL_TRUE, 0,
      Sizeof.cl_int * 2 * m, Pointer.to(nv1v2) , 0, null, null)

    println("nv1v2 array: " + jArrays.toString(nv1v2))


    val V1 : mutable.ListBuffer[Int] = mutable.ListBuffer.empty[Int]
    V1x.copyToBuffer(V1)
    val V1Boolean : mutable.ListBuffer[Boolean] = mutable.ListBuffer.empty[Boolean]
    val V2Boolean : mutable.ListBuffer[Boolean] = mutable.ListBuffer.empty[Boolean]
    val V2 : mutable.ListBuffer[Int] = mutable.ListBuffer.empty[Int]
    V2x.copyToBuffer(V2)

    println("array v1" + V1)
    println("array v2" + V2)
    for(_ <- 0 to V1x.length) {
      V1Boolean.+=(false)
      V2Boolean.+=(false)
    }

    nv1v2.foreach(f => {
      val Nf : Int = NArray(f - 1)
      var V1Index : Int = 0
      var V2Index : Int = 0
      breakable {
        for(i <- V1.indices){
          if(V1(i) == Nf && !V1Boolean(i)){
            V1Index = i
            break
          }
          V1Index = i
        }
      }

      breakable {
        for(i <- V2.indices){
          if(V2(i) == Nf && !V2Boolean(i)){
            V2Index = i
            break
          }
          V2Index = i
        }
      }

      if(V1Index <= V2Index){
        V1(V1Index) = f
        V1Boolean(V1Index) = true
      } else {
        V2(V2Index) = f
        V2Boolean(V2Index) = true
      }
    })

    //println(V1)
    //println(V2)
    var filterListV : mutable.ListBuffer[Int] = mutable.ListBuffer.empty[Int]

    filterListV = filterListV ++ V1 ++ V2
    var firstNode : Int = 0

    filterListV.groupBy(identity).mapValues(_.size).foreach(f => if(f._2 == 1) firstNode = f._1)
    for(i <- 0 to V1x.length) {
      V1Boolean(i) = false
    }

    //println(V1Boolean)
    var finalList : mutable.ListBuffer[Int] = mutable.ListBuffer.empty[Int]
    finalList += firstNode
    while(finalList.length < NArray.length) {
      breakable {
        for (i <- V1.indices) {
          if (V1(i) == firstNode && !V1Boolean(i)) {
            finalList += V2(i)
            V1Boolean(i) = true
            firstNode = V2(i)
            break
          } else if (V2(i) == firstNode && !V1Boolean(i)) {
            finalList += V1(i)
            V1Boolean(i) = true
            firstNode = V1(i)
            break
          }
        }
      }
    }


    println("Euler tour" + finalList)
  }

  def destroy(): Unit ={
    clReleaseKernel(kernel)
    clReleaseProgram(program)
  }
}
