package algorithms

import java.util.{Arrays => jArrays}

import org.graphstream.graph.{Edge, Node}

import scala.collection.mutable
import scala.collection.immutable
import scala.collection.immutable.ListMap

object GraphToGpuArray {

  private var nodesHashMap: scala.collection.mutable.Map[String, Int] = _
  private var edgesHashMap: scala.collection.mutable.Map[String, List[Int]] = _
  private var changed: Boolean = true
  private var nodesArray, edgesArray: Array[Int] = _
  private var V1Array, V2Array: Array[Int] = _

  def Init(): Unit = {
    initEdgeArray()
    initNodeArray()
  }

  private def buildGpuArrays(): Unit = {
    buildGpuArraysV2()
    val nodeList : mutable.ListBuffer[Int] = mutable.ListBuffer.empty[Int]
    val edgeList : mutable.ListBuffer[Int] = mutable.ListBuffer.empty[Int]

    var actualNumberOfEdges : Int = 0
    for ((k, _) <- nodesHashMap) {
      val nodeEdgeList : List[Int] = edgesHashMap(k)

      nodeList += actualNumberOfEdges
      actualNumberOfEdges = actualNumberOfEdges + nodeEdgeList.length
      nodeEdgeList.foreach(f => edgeList += f)
    }

    nodesArray = nodeList.toArray
    edgesArray = edgeList.toArray
    changed = false
  }

  private def buildGpuArraysV2() : Unit = {
    val V1 : mutable.ListBuffer[Int] = mutable.ListBuffer.empty[Int]
    val V2 : mutable.ListBuffer[Int] = mutable.ListBuffer.empty[Int]
    val visited : mutable.ListBuffer[(Int, Int)] = mutable.ListBuffer.empty[(Int, Int)]

    val hashList : immutable.ListMap[String, Int] = ListMap(nodesHashMap.toSeq.sortBy(_._1):_*)
    for ((k, _) <- hashList) {
      var nodeEdgeList : List[Int] = edgesHashMap(k)
      nodeEdgeList =  nodeEdgeList.sorted

      val vertex1 : Int = k.toInt
      nodeEdgeList.foreach(vertex2 => {
        if(!visited.exists(p => (p._1.equals(vertex1) && p._2.equals(vertex2))
          || (p._1.equals(vertex2) && p._2.equals(vertex1)))){
          visited += ((vertex1, vertex2))
          V1 += vertex1
          V2 += vertex2
        }
      })
    }

    V1Array = V1.toArray
    V2Array = V2.toArray

    //println(jArrays.toString(V1Array))
    //println(jArrays.toString(V2Array))
  }

  def getGpuNodeArray: Array[Int] = {
    if (changed) buildGpuArrays()
    nodesArray
  }

  def getGpuEdgeArray: Array[Int] = {
    if (changed) buildGpuArrays()
    edgesArray
  }

  def getGpuV1Array : Array[Int] = {
    if (changed) buildGpuArrays()
    V1Array
  }

  def getGpuV2Array : Array[Int] = {
    if (changed) buildGpuArrays()
    V2Array
  }

  private def initNodeArray(): Unit = nodesHashMap = scala.collection.mutable.Map.empty[String, Int]
  private def initEdgeArray(): Unit = edgesHashMap = scala.collection.mutable.Map.empty[String, List[Int]]

  def addNode(node : Node): Unit = {
    val nodeID : Integer = node.getAttribute("ui.label")
    nodesHashMap += (nodeID + "") -> nodeID
    edgesHashMap += (nodeID + "") -> List[Int]()
    changed = true
  }

  def addEdge(edge : Edge): Unit = {
    val nodeA = edge.getNode0[Node]
    val nodeB = edge.getNode1[Node]

    val nodeAID : Integer = nodeA.getAttribute("ui.label")
    val nodeBID : Integer = nodeB.getAttribute("ui.label")

    val edgeListA : List[Int] = edgesHashMap(nodeAID + "")
    val edgeListB : List[Int] = edgesHashMap(nodeBID + "")

    edgesHashMap += (nodeAID + "") -> (edgeListA :+ nodeBID.intValue())
    edgesHashMap += (nodeBID + "") -> (edgeListB :+ nodeAID.intValue())
    changed = true
  }

  def removeNode(node : Node) : Unit = {
    val nodeID : Integer = node.getAttribute("ui.label")
    nodesHashMap.retain((_, v) => !v.equals(nodeID))
    edgesHashMap.retain((k,_) => !k.equals(nodeID + ""))
    for ((k, v) <- edgesHashMap) {
      edgesHashMap(k) = v.filter(value => value != nodeID)
    }
    changed = true
  }

}
