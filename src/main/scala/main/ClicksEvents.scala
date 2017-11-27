package main

import java.awt.event.MouseEvent

import algorithms.GraphToGpuArray
import main.GraphUtils.{addEdge, getEdge, getNode}
import org.graphstream.graph.implementations.SingleGraph
import org.graphstream.ui.graphicGraph.{GraphicElement, GraphicGraph}
import org.graphstream.ui.layout.springbox.SpringBox
import org.graphstream.ui.swingViewer.util.MouseManager
import org.graphstream.ui.swingViewer.{View, Viewer}

class ClicksEvents(graph: SingleGraph, viewer : Viewer, layout: SpringBox, graphic: GraphicGraph, view: View) extends MouseManager(graphic, view){

  override def mouseClicked(e: MouseEvent): Unit = {
    super.mouseClicked(e)
    val currentElement = view.findNodeOrSpriteAt(e.getX, e.getY)
    layout.shake()
    if (currentElement == null) {
      val gu = view.getCamera.transformPxToGu(e.getX, e.getY)
      layout.shake()
      val nodeID = GraphUtils.nodeLabel
      graph.addNode(nodeID + "")
      GraphUtils.nodeLabel = nodeID + 1
      val node = getNode(graph, nodeID + "")
      node.addAttribute("x", double2Double(gu.x))
      node.addAttribute("y", double2Double(gu.y))
      node.addAttribute("ui.label", int2Integer(nodeID))
      GraphToGpuArray.addNode(node)
    }

  }

  override def mouseButtonPressOnElement(element: GraphicElement, event: MouseEvent): Unit = {
    super.mouseButtonPressOnElement(element, event)
    //println("apreté")
  }

  override def mouseButtonReleaseOffElement(element: GraphicElement, event: MouseEvent): Unit = {
    curElement = view.findNodeOrSpriteAt(event.getX, event.getY)
    if (curElement != null) {
      layout.shake()
      val edgeID = GraphUtils.edgeLabel
      addEdge(graph, edgeID + "", curElement.getId, element.getId)
      val edge = getEdge(graph, edgeID + "")
      if (edge != null) {
        edge.setAttribute("ui.style", "fill-color: black;")
        edge.setAttribute("ui.label", "" + edgeID)
        GraphToGpuArray.addEdge(edge)
        GraphUtils.edgeLabel = edgeID + 1
        //edge.setAttribute("ui.label", "hola")
      }
    }

    super.mouseButtonReleaseOffElement(element, event)
    //println("solté")
  }

  override def mouseDragged(event: MouseEvent): Unit = super.mouseDragged(event)

  override def mousePressed(event: MouseEvent): Unit = {
    super.mousePressed(event)
  }

  override def mouseMoved(e: MouseEvent): Unit = super.mouseMoved(e)
}
