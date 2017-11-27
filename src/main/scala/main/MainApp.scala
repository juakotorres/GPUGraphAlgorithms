package main

import algorithms.GraphToGpuArray
import gpu.utils.OpenCLInit
/**
  * Aquí nos encargaremos de inicializar la aplicación.
  */
object MainApp {

  def main(args: Array[String]): Unit = {
    OpenCLInit.openclInit()
    GraphToGpuArray.Init()
    GraphUtils.Init()
  }
}



