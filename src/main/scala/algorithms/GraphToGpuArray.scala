package algorithms


import org.graphstream.graph.{Edge, Node}

import scala.collection.mutable

object GraphToGpuArray {

  private var nodesHashMap: scala.collection.mutable.Map[String, Int] = _
  private var edgesHashMap: scala.collection.mutable.Map[String, List[Int]] = _
  private var changed: Boolean = true
  private var nodesArray: Array[Int] = _
  private var edgesArray: Array[Int] = _

  def Init(): Unit = {
    initEdgeArray()
    initNodeArray()
  }

  def buildGpuArrays(): Unit = {
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

  def getGpuNodeArray: Array[Int] = {
    if (changed) buildGpuArrays()
    nodesArray
  }

  def getGpuEdgeArray: Array[Int] = {
    if (changed) buildGpuArrays()
    edgesArray
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
