``` mermaid
flowchart LR
    %% Controller (Jetson) and its control lines  
    JETSON[Jetson Orin NX / J401 GPIO]  
    DIR1_CTRL["GPIO → DIR1"]  
    DIR2_CTRL["GPIO → DIR2"]  
    PWM_CTRL["PWM / ENABLE"]  

    %% Power supply to driver  
    MOTOR_PWR["External 12 V Motor Power"]  

    %% Motor driver (H-bridge) and actuator outputs  
    HBRIDGE[H-bridge Motor Driver]  
    ACT_PLUS["Actuator + wire"]  
    ACT_MINUS["Actuator – wire"]  

    %% Connections from Jetson to H-bridge control inputs  
    JETSON --> DIR1_CTRL --> HBRIDGE  
    JETSON --> DIR2_CTRL --> HBRIDGE  
    JETSON --> PWM_CTRL --> HBRIDGE  

    %% Power line to H-bridge  
    MOTOR_PWR --> HBRIDGE  

    %% H-bridge → actuator wires  
    HBRIDGE --> ACT_PLUS  
    HBRIDGE --> ACT_MINUS
```
