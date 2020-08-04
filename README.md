# Heat Transfer in a Metal Plate

### 1. Summary

  In this project I implemented a GPU-accelerated application that simulates the heat propagation on a conductive metal plate. The progress of the simulation is displayed through a 2D thermal imaging animation. 
  
### 2. Physical Context

  Consider an environment which contains a 2D heat-conductive metal plate. The plate has a initial temperature and is surrounded by air. Then we can introduce a small constant heat source on the metal plate and observe how the heat progressively spreads on the surface.
  
### 3. Simulation Details

  The simulation parameters must be written in the config.in file, in the root directory.
  You can specify the platform of your preffered gpu, the widh and height in pixels of the plate, its initial temperature, the air's temperature and the source temperature.
  
  platform:Intel<br/>
  width:640<br/>
  height:480<br/>
  initial_temp:30.0<br/>
  air_temp:40.0<br/> 
  point_temp:5500.0<br/>
  
  The heat source can be moved using the mouse. Also some initial parameters, such as the point temperature and the air temperature, can be changed during the simulation.
  To proove that this simulation runs better using a GPU, I added the possibility to balance the load of computation between the GPU and the CPU. This can be done using the slider labeled "f" in the simulation. When f is equal to 100, the simulation is ran only on the GPU, otherwise the CPU will use 4 threads to make some calculations aswell.
  
  

  
