#include <SPI.h>
#include <Adafruit_INA219.h>
#include <SD.h>
#define MUX_Address 0x70

//Adafruit_INA219 ina219_A1(0x40);
Adafruit_INA219 ina219_B1(0x41);
Adafruit_INA219 ina219_C1(0x44);
Adafruit_INA219 ina219_A2(0x40);
Adafruit_INA219 ina219_B2(0x41);
Adafruit_INA219 ina219_C2(0x44);


File PowConsFile;
File debugFile;
//File TimingFile;
const int chipSelect = 4;
//int alarm = 3;
int alarm = 9;


// To run the I2C MUX.
void tcaselect ( uint8_t i2c_bus) {
  if (i2c_bus > 7 ) return;
  Wire.beginTransmission(MUX_Address);
  Wire.write(1 << i2c_bus);
  Wire.endTransmission();
  //delay(10);
}



void setup() {
  Serial.begin(9600);
  //  // To use a slightly lower 32V, 1A range (higher precision on amps):
  //  ina219.setCalibration_32V_1A();
  //  // Or to use a lower 16V, 400mA range (higher precision on volts and amps):
  //  //ina219.setCalibration_16V_400mA();

  tcaselect(1);
//  ina219_A1.begin();  ina219_A1.setCalibration_16V_400mA();
  ina219_B1.begin();  ina219_B1.setCalibration_16V_400mA();
  ina219_C1.begin();  ina219_C1.setCalibration_16V_400mA();


  tcaselect(2);
  ina219_A2.begin();  ina219_A2.setCalibration_16V_400mA();
  ina219_B2.begin();  ina219_B2.setCalibration_16V_400mA();
  ina219_C2.begin();  ina219_C2.setCalibration_16V_400mA();


  pinMode(chipSelect, OUTPUT);
  pinMode(alarm, OUTPUT);
  
//  //''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  // see if the card is present and can be initialized:
  if (!SD.begin(chipSelect)) {
    Serial.println("Card failed, or not present");
    // don't do anything more:
    while (!SD.begin(chipSelect)) {
      digitalWrite (alarm, 220);
      delay(50);
      digitalWrite (alarm, 0);
      delay(500);
    }
  }
  PowConsFile = SD.open("Data.txt", FILE_WRITE);
  if (PowConsFile) {
    PowConsFile.println("Starting a new session:");
    PowConsFile.println("");
    PowConsFile.close();
  } else {
    Serial.println("couldn't write to SD");
  }

    
//  //''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  digitalWrite (alarm, 0);

  delay(5000); // To allow for the chips to power on and begin operating
}




//float A1_power_mW;
float B1_power_mW;
float C1_power_mW;
float A2_power_mW;
float B2_power_mW;
float C2_power_mW;



void loop() {



// ///////////////Get new datapoint////////////////

  tcaselect(1);

//  A1_power_mW = ina219_A1.getPower_mW();
  B1_power_mW = ina219_B1.getPower_mW();
  C1_power_mW = ina219_C1.getPower_mW();

  tcaselect(2);
  A2_power_mW = ina219_A2.getPower_mW();
  B2_power_mW = ina219_B2.getPower_mW();
  C2_power_mW = ina219_C2.getPower_mW();


   
//////////////////////////////////////////////////////


  //  TimingFile = SD.open("Timefile.txt", FILE_WRITE);
  //  if(TimingFile){
  //    //TimingFile.print(t1);
  //    //TimingFile.print(",  ");
  //    //TimingFile.print(t2);
  //    //TimingFile.print(",  ");
  //    //TimingFile.print(t2 - t1);
  //    //TimingFile.print(",  ");
  //    //TimingFile.print(t3 - t2);
  //    //TimingFile.print(",  ");
  //    TimingFile.println(t4 - t1);
  //  }
  //  TimingFile.close();


  //  // see if the card is present and can be initialized:
  //  if (!SD.begin(chipSelect)) {
  //    Serial.println("Card failed, or not present");
  //    // don't do anything more:
  //    while (!SD.begin(chipSelect)){
  //      digitalWrite (alarm, 220);
  //      delay(50);
  //      digitalWrite (alarm, 0);
  //      delay(500);
  //      }
  //  }


// ///////////////Save info to .txt files////////////////

  PowConsFile = SD.open("Data.txt", FILE_WRITE);
  if (PowConsFile) {
//    PowConsFile.print(A1_power_mW);
//    PowConsFile.print(",  ");
    PowConsFile.print(B1_power_mW);
    PowConsFile.print(",  ");
    PowConsFile.print(C1_power_mW);
    PowConsFile.print(",  ");
    PowConsFile.print(A2_power_mW);
    PowConsFile.print(",  ");
    PowConsFile.print(B2_power_mW);
    PowConsFile.print(",  ");
    PowConsFile.print(C2_power_mW);
    PowConsFile.println("");
    PowConsFile.close();
  } else {
    Serial.println("couldn't write to SD");
  }

}
