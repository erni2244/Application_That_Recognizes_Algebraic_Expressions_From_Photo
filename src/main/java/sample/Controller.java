package sample;

import DeepLearning.Learn;
import DeepLearning.Recognition;
import PictureTransform.Segmentation;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class Controller {
    Recognition recognition;
    Learn learn;
    Segmentation segmentation;

    @FXML
    ImageView image_view;

    @FXML
    Label odpText;

    @FXML
    public void initialize(){
        learn=new Learn();
        String s="C:/Users/lasek/Desktop/test.png"; //path to file to recognition
        FileInputStream input = null;
        try {
            input = new FileInputStream(s);
            Image image = new Image(input);
            image_view.setImage(image);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        segmentation=new Segmentation(s);
        recognition=new Recognition();
    }



    public void Learn_click(ActionEvent actionEvent) {
        try {
            learn.fun();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void Detect_click(ActionEvent actionEvent) {
        //segmentation.extract_grid();
        //segmentation.f12();
        //segmentation.movepicture();
        String odpowiedz;
        try{
            //segmentation.separate();
            //recognition.recogniseHaar(segmentation.separate());
            odpowiedz=recognition.recognise(segmentation.separate());
            odpText.setText(odpowiedz);
        }catch (Exception e){
            System.out.println("Error "+e);
        }

    }



}
