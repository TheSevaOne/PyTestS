import pytest
import app.driver as app 

def test_ini():
    classesFile,modelConfiguration,modelWeights,obj= app._input('C:\\Users\\Seva\\Desktop\\videoprocessing testing\\app\\ini\\or-helmet_detection.ini')
    assert obj=='Helmet'

def test_ini_fail():
      classesFile,modelConfiguration,modelWeights,obj= app._input('C:\\Users\\Seva\\Desktop\\videoprocessing testing\\app\\ini\\or-human_detection.ini')
      assert obj!='Helmet'

def load_image(image):
     pass
    