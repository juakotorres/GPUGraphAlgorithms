package main

import org.graphstream.graph.implementations.SingleGraph
import org.graphstream.graph.{Edge, Graph, Node}
import org.graphstream.ui.layout.springbox.SpringBox
import org.graphstream.ui.swingViewer.Viewer

object GraphUtils {

  var nodeLabel : Int = 0
  var edgeLabel : Int = 0
  var arrowsOn : Boolean = false
  val styleSheet : String = "node {"+
    " fill-color: black;"+
    " size: 20px;" +
    " text-background-mode: rounded-box;" +
    " text-background-color: blue;" +
    " text-size: 20px;"+
    " stroke-mode: plain;"+
    " stroke-color: black;"+
    " stroke-width: 1px;"+
    "}"+
    "node.important {"+
    " fill-color: red;"+
    " size: 30px;"+
    "}"+
    "edge {"+
    " fill-color: black;" +
    " text-size: 20px;" +
    " size: 3px;" +
    "}"

  def Init() : Unit = {

    val graph = new SingleGraph("Tutorial 1")
    // Con esta línea podemos crear arista que contengan nodos inexistentes en el grafo
    graph.setStrict(false)
    graph.setAutoCreate(true)

    // Le da color al nodo.
    //a.addAttribute("ui.style", "fill-color: rgb(0,100,255);")

    graph.addAttribute("ui.stylesheet", styleSheet)

    // Mostramos el grafo.
    val viewer = graph.display(false)

    val layout = new SpringBox()
    viewer.enableAutoLayout(layout)

    val view = viewer.getDefaultView

    viewer.setCloseFramePolicy(Viewer.CloseFramePolicy.HIDE_ONLY)
    view.getKeyListeners.foreach(f => view.removeKeyListener(f))
    view.getMouseListeners.foreach(f => view.removeMouseListener(f))
    view.getMouseMotionListeners.foreach(f => view.removeMouseMotionListener(f))

    view.addKeyListener(new KeyBoardEvents(graph, view))
    view.addMouseListener(new ClicksEvents(graph, viewer, layout, viewer.getGraphicGraph, view))
  }


  /**
    * Add node to the graph.
    */
  def addNode(graph: Graph, name: String): Unit = {
    graph.addNode(name)
  }

  /**
    * Add edge to graph.
    */
  def addEdge(graph: Graph, name: String, nodeA: String, nodeB: String) : Unit = {
    graph.addEdge(name, nodeA, nodeB, arrowsOn)
  }

  def getNode(graph: Graph, str: String) : Node = {
    graph.getNode(str)
  }

  def getEdge(graph: Graph, str: String) : Edge = {
    graph.getEdge(str)
  }
}
