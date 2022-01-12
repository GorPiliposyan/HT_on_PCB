/* Set the type of sleep mode we want. Can be one of (in order of power saving):
        SLEEP_MODE_IDLE (Timer 0 will wake up every millisecond to keep millis running)
        SLEEP_MODE_ADC
        SLEEP_MODE_PWR_SAVE (TIMER 2 keeps running)
        SLEEP_MODE_EXT_STANDBY
        SLEEP_MODE_STANDBY (Oscillator keeps running, makes for faster wake-up)
        SLEEP_MODE_PWR_DOWN (Deep sleep)
    */
    
#include <avr/sleep.h>
#include <avr/wdt.h>
#define wakePin 2   // when low, makes 328P wake up; must be an interrupt pin (2 or 3 on ATMEGA328P)

volatile int sleepCnt = 0;

void setup() {
//    delay(1360000);
//    pinMode(wakePin, INPUT_PULLUP);


//    set_sleep_mode (SLEEP_MODE_ADC);  //16mW-40mW
//    set_sleep_mode (SLEEP_MODE_PWR_SAVE); //6mW-10mW
    
//    set_sleep_mode (SLEEP_MODE_PWR_DOWN); //~0mW
//    sleep_enable();
//    noInterrupts();
//    attachInterrupt(digitalPinToInterrupt(wakePin), sleepISR, LOW);
//    interrupts();
//    sleep_cpu();
    
//    set_sleep_mode (SLEEP_MODE_EXT_STANDBY); //6mW-10mW
//    sleep_enable();
//    noInterrupts();
//    attachInterrupt(digitalPinToInterrupt(wakePin), sleepISR, LOW);
//    interrupts();
//    sleep_cpu();

}

//int counter = 0;

void loop() {
  
    //Reset sleep counter
  sleepCnt = 0;
  set_sleep_mode(SLEEP_MODE_ADC);  //16mW-40mW
  sleep_enable();
  
  while(sleepCnt < 125) {
    noInterrupts();
    MCUSR = 0; // allow changes, disable reset
    WDTCSR = bit (WDCE) | bit(WDE); // set interrupt mode and an interval
    WDTCSR = bit (WDIE) | bit(WDP2) | bit(WDP1) | bit(WDP0); // set WDIE, and 2 second delay
    wdt_reset();
    interrupts();
    sleep_cpu();
  }
  wdt_reset();
  // Prevent sleep mode, so we don't enter it again, except deliberately, by code
  sleep_disable();

  //Reset sleep counter
  sleepCnt = 0;
  wdt_reset();
  set_sleep_mode(SLEEP_MODE_PWR_DOWN); //~0mW
  sleep_enable();
  
  while(sleepCnt < 125){
    noInterrupts();
    MCUSR = 0; // allow changes, disable reset
    WDTCSR = bit (WDCE) | bit(WDE); // set interrupt mode and an interval
    WDTCSR = bit (WDIE) | bit(WDP2) | bit(WDP1) | bit(WDP0); // set WDIE, and 2 second delay
    wdt_reset();
    interrupts();
    sleep_cpu();
  }
  
  // Prevent sleep mode, so we don't enter it again, except deliberately, by code
  sleep_disable();


}


// When wakePin is brought LOW this interrupt is
// triggered FIRST (even in PWR_DOWN sleep)

//void sleepISR() {
//  
//  // Prevent sleep mode, so we don't enter it 
//  // again, except deliberately, by code
//  
//  sleep_disable();
//
//  // Detach the interrupt that brought us out of sleep
//  detachInterrupt(digitalPinToInterrupt(wakePin));
//
//  // Now we continue running the main Loop() just after we went to sleep
//}

// When WatchDog timer causes uC to wake, it comes here
ISR (WDT_vect){
  //Turn off the watchdog timer, we don't want it to do anything (like restting this sketch)
  wdt_disable();

  //Increment the WDT interrupt count
    sleepCnt = sleepCnt + 1;

  //Now we continue running the main Loop() just after we went to sleep
}
