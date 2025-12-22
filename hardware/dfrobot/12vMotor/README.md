Website: https://www.dfrobot.com/product-633.html
Wiki: https://wiki.dfrobot.com/12V_DC_Motor_251rpm_w_Encoder__SKU__FIT0186_


SKU:FIT0186
FIT0186 Metal DC Geared Motor w/Encoder - 12V 251RPM 18Kg.cm

Introduction
This is a Gear Motor w/Encoder, model No.GB37Y3530-12V-251R. It is a powerful 12V motor with a 43.8:1 metal gearbox and an integrated quadrature encoder that provides a resolution of 16 counts per revolution of the motor shaft, which corresponds to 700 counts per revolution of the gearbox’s output shaft. These units have a 0.61" long, 6 mm-diameter D- shaped output shaft. This motor is intended for use at 12V, though the motor can begin rotating at voltages as low as 1V. The face plate has six mounting holes evenly spaced around the outer edge threaded for M3 screws. These mounting holes form a regular hexagon and the centers of neighboring holes are 15.5 mm apart. This motor is an ideal option for your mobile robot project.

Warning: Do not screw too far into the mounting holes as the screws can hit the gears. Manufacturer recommends screwing no further than 3mm (1/8") into the screw hole.

Specification
Gear ratio: 43.8:1
No-load speed: 251 10% RPM
No-load current: 350 mA
Start Voltage: 1.0 V
Stall Torque: 18 Kg.com
Stall Current: 7 A
Insulation resistance: 20 M Ω
EncoderOperating Voltage: 5 V
Encoder type: Hall
Encoder Resolution: 16CPR(motor shaft)/700CPR(gearbox shaft)
Weight: 205g
Encoder Diagram
Diagram for UNO

FIT0186 Metal DC Geared Motor w/Encoder - 12V 251RPM 18Kg.cm Encoder Diagram
Interrupt Port with Different Board

Notcie: attachInterrupt()

FIT0186 Metal DC Geared Motor w/Encoder - 12V 251RPM 18Kg.cm Encoder Diagram
For example,with UNO board, you want to use interrupt port 0(int.0). You should connect digital pin 2 with the board. So, the following code is only used in UNO and Mega2560. If you want to use Leonardo or Romeo, you should change digital pin 3 instead of digital pin 2.

See the link for detail http://arduino.cc/en/Reference/AttachInterrupt

Encoder Sample Code
```
//The sample code for driving one way motor encoder
const byte encoder0pinA = 2;//A pin -> the interrupt pin 0
const byte encoder0pinB = 4;//B pin -> the digital pin 4
byte encoder0PinALast;
int duration;//the number of the pulses
boolean Direction;//the rotation direction


void setup()
{
  Serial.begin(57600);//Initialize the serial port
  EncoderInit();//Initialize the module
}

void loop()
{
  Serial.print("Pulse:");
  Serial.println(duration);
  duration = 0;
  delay(100);
}

void EncoderInit()
{
  Direction = true;//default -> Forward
  pinMode(encoder0pinB,INPUT);
  attachInterrupt(0, wheelSpeed, CHANGE);
}

void wheelSpeed()
{
  int Lstate = digitalRead(encoder0pinA);
  if((encoder0PinALast == LOW) && Lstate==HIGH)
  {
    int val = digitalRead(encoder0pinB);
    if(val == LOW && Direction)
    {
      Direction = false; //Reverse
    }
    else if(val == HIGH && !Direction)
    {
      Direction = true;  //Forward
    }
  }
  encoder0PinALast = Lstate;

  if(!Direction)  duration  ;
  else  duration--;
} 
```