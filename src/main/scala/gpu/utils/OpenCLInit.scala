package gpu.utils

import org.jocl.CL._
import org.jocl._

import scala.Array.ofDim

/**
  * Contiene todos los elementos básicos necesarios para ejecutar en GPU con openCL.
  */
object OpenCLInit {

  // The platform, device type and device number
  // that will be used
  private val platformIndex = 0
  private val deviceType : Long = CL_DEVICE_TYPE_ALL
  private val deviceIndex = 0
  private var platform : cl_platform_id = _
  private var device : cl_device_id = _
  private var context : cl_context = _
  private var commandQueue : cl_command_queue = _

  def openclInit(): Unit = {

    // Tomamos la plataforma openCL en la máquina.
    platform = {

      // Obtenemos los números de plataformas existentes
      val numPlatformsArray = Array(0)
      clGetPlatformIDs(0, null, numPlatformsArray)
      val numPlatforms = numPlatformsArray(0)

      // Obtenemos el identificador
      val platforms = ofDim[cl_platform_id](numPlatforms)
      clGetPlatformIDs(platforms.length, platforms, null)
      platforms(platformIndex)
    }

    // Inicializamos propiedades del contexto, agregamos plataforma al contexto.
    val contextProperties = new cl_context_properties()
    contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform)

    device = {
      // Obtenemos los dispositivos disponibles
      val numDevicesArray = Array(0)
      clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
      val numDevices = numDevicesArray(0)

      // Obtenemos el id del dispositivo
      val devices = ofDim[cl_device_id](numDevices)
      clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
      devices(deviceIndex)
    }

    // creamos un contexto para el dispositivo elegido.
    context = clCreateContext(contextProperties, 1, Array(device),
      null, null, null)

    // Create a command-queue for the selected device
    commandQueue = clCreateCommandQueueWithProperties(context, device, null, null)
  }

  def getPlatform: cl_platform_id = platform
  def getContext: cl_context = context
  def getDevice: cl_device_id = device
  def getCommandQueue : cl_command_queue = commandQueue

  def destroy(): Unit ={
    clReleaseCommandQueue(commandQueue)
    clReleaseContext(context)
  }

}
