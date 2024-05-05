import sumolib
import traci


sumoBinary = sumolib.checkBinary('sumo-gui')
traci.start([sumoBinary, "-c", 'C:\Program Files (x86)\Eclipse\Sumo\Project\Simulation.sumocfg']) 
i=0

for step in range(3600):
    traci.simulationStep()
    list=traci.trafficlight.getIDList()
    trafficLight1=list[0]
    i=i+1
    vehicle_count = traci.edge.getLastStepVehicleNumber("fromLeft")
    vehicle_count2 = traci.edge.getLastStepVehicleNumber("fromRight")
    vehicle_count3 = traci.edge.getLastStepVehicleNumber("fromTop")
    vehicle_count4 = traci.edge.getLastStepVehicleNumber("fromBottom")

    
    if (i==5):
        print(traci.trafficlight.getRedYellowGreenState(trafficLight1))
        print(f"{vehicle_count} vehicles in the left edge")
        print(f"{vehicle_count2} vehicles in the right edge")
        print(f"{vehicle_count3} vehicles in the top edge")
        print(f"{vehicle_count4} vehicles in the bottom edge")
        totalCars=vehicle_count+vehicle_count2+vehicle_count3+vehicle_count4
        if(totalCars>=100):
            traci.close()
        
        i=0
    

   



traci.close()
