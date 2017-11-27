package main

import java.awt.event.KeyEvent

import algorithms.{EulerTour, GraphToGpuArray}
import org.graphstream.graph.Graph
import org.graphstream.ui.swingViewer.View
import org.graphstream.ui.swingViewer.util.ShortcutManager

class KeyBoardEvents(graph: Graph, view : View) extends ShortcutManager(view){


  override def keyPressed(event: KeyEvent): Unit = {
    super.keyPressed(event)

    if (event.getKeyCode == KeyEvent.VK_E) { // Calcular el camino euleriano
      EulerTour.EulerTourInit(GraphToGpuArray.getGpuNodeArray, GraphToGpuArray.getGpuEdgeArray)
      if(EulerTour.eulerTourDetection())
        println("Tiene camino euleriano!!")
    } else if (event.getKeyCode == KeyEvent.VK_R) { // Eliminar nodo del grafo.
      val position = this.view.getMousePosition(true)
      val currentElement = view.findNodeOrSpriteAt(position.getX, position.getY)
      if (currentElement != null){
        GraphToGpuArray.removeNode(graph.getNode(currentElement.getId))
        graph.removeNode(currentElement.getId)
      }

    }
  }

  override def keyReleased(event: KeyEvent): Unit = {
    super.keyReleased(event)
  }
}
