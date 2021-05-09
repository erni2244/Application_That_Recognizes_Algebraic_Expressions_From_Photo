package sample;

import DeepLearning.Learn;
import DeepLearning.Recognition;
import PictureTransform.Segmentation;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.ChoiceBox;
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
    static final String[] typy_sieci={"Sieć konwolucyjna","Sieć MLP","Maszyna SVM"};
    @FXML
    ImageView image_view;

    @FXML
    Label odpText;

    @FXML
    ChoiceBox<String> select_type_network;

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
        make_ChoiceBox_network_type();
        segmentation=new Segmentation(s);
        recognition=new Recognition();
    }

    private void make_ChoiceBox_network_type(){
        select_type_network.getItems().add(typy_sieci[0]);
        select_type_network.getItems().add(typy_sieci[1]);
        select_type_network.getItems().add(typy_sieci[2]);
    }


    public void Learn_click(ActionEvent actionEvent) {
        try {

            if(select_type_network.getValue()==typy_sieci[0])
                learn.start_learn_convalution();
            else if(select_type_network.getValue()==typy_sieci[1])
                learn.start_learn_MLP();
            else if(select_type_network.getValue()==typy_sieci[2])
                learn.start_learn_SVM();
            else
                odpText.setText("Proszę wybrać typ sieci");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void Detect_click(ActionEvent actionEvent) {
        //segmentation.extract_grid();
        //segmentation.movepicture();
        String odpowiedz;
        try{
            odpowiedz="Proszę wybrać typ sieci";
            if(select_type_network.getValue()==typy_sieci[0])
                odpowiedz=recognition.recognise(segmentation.separate());
            if(select_type_network.getValue()==typy_sieci[1])
                odpowiedz=recognition.recognise1();
            if(select_type_network.getValue()==typy_sieci[2])
                odpowiedz=recognition.recogniseHaar(segmentation.separate());
            //segmentation.separate();
            //odpowiedz=recognition.recogniseHaar(segmentation.separate());
            //odpowiedz=recognition.recognise(segmentation.separate());
            //odpowiedz=recognition.recognise1();
            odpText.setText(odpowiedz);
        }catch (Exception e){
            System.out.println("Error "+e);
        }

    }



}
