import time
import threading

class InputTextWidget:

    def __init__(self,inputwidget) -> None:
        self.__inputWidget = inputwidget

    def input(self,isdigit = True):
        datacounttext = self.__inputWidget.text
        if datacounttext == "" or not datacounttext.isdigit():
            return None

        datacount = int(datacounttext)

        return datacount
    

class ButtonCoolDown:

    def __init__(self,btn,timecount = 2,count = 1) -> None:
        self.__btn = btn
        self.__default_text = btn.text

        self.__timecountDefault = timecount * count
        self.__timecount = timecount * count
        
        self.start()

    
    def start(self):
        self.disable()
        threading.Thread(target=self.__threadCooldownText).start()

    def disable(self):
        self.__btn.disabled = True
        self.__loop = False
    def undisable(self):
        self.__btn.disabled = False
    def __threadCooldownText(self):
        while True:
            self.__btn.text = "%.2f" % self.__timecount
            if self.__timecount <= 0:
                break
               
            time.sleep(0.01)
            self.__timecount -= 0.01
        self.__btn.text = self.__default_text
        self.undisable()

        