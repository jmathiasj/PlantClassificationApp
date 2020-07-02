package com.example.flowerclassification;


import android.Manifest;
import android.annotation.TargetApi;
import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.annotation.RequiresApi;
import android.support.design.widget.CollapsingToolbarLayout;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;
import android.widget.ImageView;
import android.graphics.Bitmap;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import android.graphics.BitmapFactory;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import android.os.AsyncTask;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.common.modeldownload.FirebaseLocalModel;
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager;
import com.google.firebase.ml.common.modeldownload.FirebaseRemoteModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;


public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    //PATH TO OUR MODEL FILE AND NAMES OF THE INPUT AND OUTPUT NODES
    private final int CAM_GAL_INTRO = 007;
    private final int CAM_IN = 111;
    private final int GAL_IN = 112;

    private String MODEL_PATH = "file:///android_asset/q_model.tflite";
    private String INPUT_NAME = "vgg16_input";
    private String OUTPUT_NAME = "dense_2/Softmax";
    private TensorFlowInferenceInterface tf;
    private Bitmap bitImg;
    private FirebaseModelInterpreter firebaseInterpreter;
    private FirebaseModelInputOutputOptions inputOutputOptions;
    private CollapsingToolbarLayout collapsingToolbarLayout;

    /**
     * An instance of the driver class to run model inference with Firebase.
     */
    private FirebaseModelInterpreter mInterpreter;
    /**
     * Data configuration of input & output data of model.
     */
    private FirebaseModelInputOutputOptions mDataOptions;

    private List<String> mLabelList;



    //ARRAY TO HOLD THE PREDICTIONS AND FLOAT VALUES TO HOLD THE IMAGE DATA
    float[] PREDICTIONS = new float[17];
    private float[][][][] floatValues;
    private int[] INPUT_SIZE = {-1,200,200,3};

    ImageView imageView;
    TextView resultView;
    Snackbar progressBar;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        collapsingToolbarLayout = findViewById(R.id.collapsingToolbar);

        ActivityCompat.requestPermissions(MainActivity.this, new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.CAMERA},CAM_GAL_INTRO);

        collapsingToolbarLayout.setTitle("Flower Identifier");
        collapsingToolbarLayout.setExpandedTitleColor(getResources().getColor(R.color.transparent));
        collapsingToolbarLayout.setCollapsedTitleTextColor(Color.rgb(255, 255, 255));

//        perm(MainActivity.this);



        try {
            mLabelList = ImageUtils.getLabels(getAssets().open("labels.json"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        BitmapFactory.Options optionsBit = new BitmapFactory.Options();
        optionsBit.inJustDecodeBounds = true;
        optionsBit.inSampleSize = 8;
        bitImg = BitmapFactory.decodeResource(getResources(), R.drawable.ic_flower, optionsBit);




        FirebaseModelDownloadConditions.Builder conditionsBuilder =
                new FirebaseModelDownloadConditions.Builder().requireWifi();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            // Enable advanced conditions on Android Nougat and newer.
            conditionsBuilder = conditionsBuilder
                    .requireCharging()
                    .requireDeviceIdle();
        }
        FirebaseModelDownloadConditions conditions = conditionsBuilder.build();

// Build a remote model source object by specifying the name you assigned the model
// when you uploaded it in the Firebase console.
        FirebaseRemoteModel cloudSource = new FirebaseRemoteModel.Builder("flower-detector")
                .enableModelUpdates(true)
                .setInitialDownloadConditions(conditions)
                .setUpdatesDownloadConditions(conditions)
                .build();
        FirebaseModelManager.getInstance().registerRemoteModel(cloudSource);

        FirebaseLocalModel localSource =
                new FirebaseLocalModel.Builder("q_model")  // Assign a name to this model
                        .setAssetFilePath("q_model.tflite")
                        .build();
        FirebaseModelManager.getInstance().registerLocalModel(localSource);



        FirebaseModelOptions options = new FirebaseModelOptions.Builder()
                .setRemoteModelName("flower-detector")
                .setLocalModelName("q_model")
                .build();
        try {

            firebaseInterpreter =
                    FirebaseModelInterpreter.getInstance(options);


            inputOutputOptions =
                    new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 200, 200, 3})
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 17})
                            .build();


        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }


        //initialize tensorflow with the AssetManager and the Model
        tf = new TensorFlowInferenceInterface(getAssets(),MODEL_PATH);


        imageView = (ImageView) findViewById(R.id.imageview);
        imageView.setTag("default");
        resultView = (TextView) findViewById(R.id.results);

        progressBar = Snackbar.make(imageView,"PROCESSING IMAGE",Snackbar.LENGTH_INDEFINITE);


        final FloatingActionButton predict = (FloatingActionButton) findViewById(R.id.predict);
        predict.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
            @Override
            public void onClick(View view) {


                    try {

                        //READ THE IMAGE FROM ASSETS FOLDER
                        getImage();

                    } catch (Exception e) {
                        e.printStackTrace();
                    }

            }
        });
    }

//    private void perm(final Context context) {
//
//        if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
//            if (ActivityCompat.shouldShowRequestPermissionRationale((Activity) context, android.Manifest.permission.READ_EXTERNAL_STORAGE)) {
//                AlertDialog.Builder alertBuilder = new AlertDialog.Builder(context);
//                alertBuilder.setCancelable(true);
//                alertBuilder.setTitle("Permission necessary");
//                alertBuilder.setMessage("External storage  read permission is necessary");
//                alertBuilder.setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
//                    @TargetApi(Build.VERSION_CODES.JELLY_BEAN)
//                    public void onClick(DialogInterface dialog, int which) {
//                        ActivityCompat.requestPermissions((Activity) context, new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 0);
//                    }
//                });
//                AlertDialog alert = alertBuilder.create();
//                alert.show();
//            } else {
//                ActivityCompat.requestPermissions((Activity) context, new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 0);
//            }
//        }
//
//
//
//
//    }


    private String userChosenTask = "";

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private void getImage() {

        final CharSequence[] items = { "Camera", "Choose from Library", "Cancel" };
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        if(imageView.getTag().equals("default") || imageView.getTag() == null){
            builder.setTitle("Select Display picture :");
        }
        else{
            builder.setTitle("Change Display picture :");
        }

        builder.setItems(items, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int item) {
                if (items[item].equals("Camera")) {
                    userChosenTask = "Camera";
                    boolean result = checkPermissions(MainActivity.this);
                    if(result)
                        cameraIntent();
                } else if (items[item].equals("Choose from Library")) {
                    userChosenTask = "Choose from Library";
                    boolean result = checkPermissions(MainActivity.this);
                    if(result)
                        galleryIntent();
                } else if (items[item].equals("Cancel")) {
                    dialog.dismiss();
                }
            }
        });
        builder.show();

    }


    public void cameraIntent(){
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, 0);
    }
    public void galleryIntent(){
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select photo"),1);
    }

    public boolean checkPermissions(final Context context){
        if(userChosenTask.equals("Choose from Library")) {
            if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                if (ActivityCompat.shouldShowRequestPermissionRationale((Activity) context, android.Manifest.permission.READ_EXTERNAL_STORAGE)) {
                    AlertDialog.Builder alertBuilder = new AlertDialog.Builder(context);
                    alertBuilder.setCancelable(true);
                    alertBuilder.setTitle("Permission necessary");
                    alertBuilder.setMessage("External storage permission is necessary");
                    alertBuilder.setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
                        @TargetApi(Build.VERSION_CODES.JELLY_BEAN)
                        public void onClick(DialogInterface dialog, int which) {
                            ActivityCompat.requestPermissions((Activity) context, new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, GAL_IN);
                        }
                    });
                    AlertDialog alert = alertBuilder.create();
                    alert.show();
                } else {
                    ActivityCompat.requestPermissions((Activity) context, new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, GAL_IN);
                }
                return false;
            } else {
                return true;
            }
        }
        else{
            if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
//                ActivityCompat.requestPermissions((Activity) context, new String[] {android.Manifest.permission.CAMERA}, 124);

                if (ActivityCompat.shouldShowRequestPermissionRationale((Activity) context, Manifest.permission.CAMERA)) {
                    AlertDialog.Builder alertBuilder = new AlertDialog.Builder(context);
                    alertBuilder.setCancelable(true);
                    alertBuilder.setTitle("Permission necessary");
                    alertBuilder.setMessage("Camera permission is necessary");
                    alertBuilder.setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
                        @TargetApi(Build.VERSION_CODES.JELLY_BEAN)
                        public void onClick(DialogInterface dialog, int which) {
                            ActivityCompat.requestPermissions((Activity) context, new String[]{Manifest.permission.CAMERA}, CAM_IN);
                        }
                    });
                    AlertDialog alert = alertBuilder.create();
                    alert.show();
                } else {
                    ActivityCompat.requestPermissions((Activity) context, new String[]{Manifest.permission.CAMERA}, CAM_IN);
                }

                return false;
            }
            else
            {
                return true;
            }

        }

    }



    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode){
            case CAM_GAL_INTRO:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                } else {
                    Toast.makeText(this, "Need permissions", Toast.LENGTH_SHORT).show();
                    ActivityCompat.requestPermissions((Activity) MainActivity.this, new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 0);
                }
                break;
            case CAM_IN:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    cameraIntent();
                } else {
                    Toast.makeText(this, "Need permissions", Toast.LENGTH_SHORT).show();
                    ActivityCompat.requestPermissions((Activity) MainActivity.this, new String[]{Manifest.permission.CAMERA}, 1);
                }
                break;
            case GAL_IN:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    galleryIntent();
                } else {
                    Toast.makeText(this, "Need permissions", Toast.LENGTH_SHORT).show();
                    ActivityCompat.requestPermissions((Activity) MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 2);
                }
                break;
            case 3:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                } else {
                    Toast.makeText(this, "Need permissions", Toast.LENGTH_SHORT).show();
                    ActivityCompat.requestPermissions((Activity) MainActivity.this, new String[]{Manifest.permission.CAMERA}, 3);
                }
                break;
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK) {
            imageView.setTag("");
            if (requestCode == 1)               //1 for gallery
                onSelectFromGalleryResult(data);
            else if (requestCode == 0)          //0 for camera
                onCaptureImageResult(data);
        }
    }

    void onSelectFromGalleryResult(Intent data){
        Bitmap bm=null;
        if (data != null) {
            try {
                bm = MediaStore.Images.Media.getBitmap(getApplicationContext().getContentResolver(), data.getData());
            } catch (IOException e) {
                e.printStackTrace();
            }
//            ByteArrayOutputStream out = new ByteArrayOutputStream();
//            bm.compress(Bitmap.CompressFormat.JPEG, 200, )

            imageView.setImageBitmap(bm);
            imageView.setTag("newImage");
            bitImg = bm;
            progressBar.show();
            predict(bitImg);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    void onCaptureImageResult(Intent data){

        Bitmap thumbnail = (Bitmap) data.getExtras().get("data");
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        Objects.requireNonNull(thumbnail).compress(Bitmap.CompressFormat.JPEG, 100, bytes);
        File destination = new File(Environment.getExternalStorageDirectory(),
                System.currentTimeMillis() + ".jpg");
        FileOutputStream fo;
        try {
            checkPermissions(getApplicationContext());
            destination.createNewFile();
            fo = new FileOutputStream(destination);
            fo.write(bytes.toByteArray());
            fo.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        imageView.setImageBitmap(thumbnail);
        imageView.setTag("newImage");
        bitImg = thumbnail;
        progressBar.show();
        predict(bitImg);

    }


    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    //FUNCTION TO COMPUTE THE MAXIMUM PREDICTION AND ITS CONFIDENCE
    public Object[] argmax(float[] array){


        int best = -1;
        float best_confidence = 0.0f;

        for(int i = 0;i < array.length;i++){

            float value = array[i];

            if (value > best_confidence){

                best_confidence = value;
                best = i;
            }
        }

        return new Object[]{best,best_confidence};


    }

    public void predict(final Bitmap bitmap){

        Bitmap resized_image = ImageUtils.processBitmap(bitmap,200);
        floatValues = ImageUtils.normalizeBitmap(resized_image,200,127.5f,1.0f);



        try {

            FirebaseModelInputs inputs = new FirebaseModelInputs.Builder()
                    .add(floatValues)  // add() as many input arrays as your model requires
                    .build();
            firebaseInterpreter.run(inputs, inputOutputOptions)
                    .addOnSuccessListener(
                            new OnSuccessListener<FirebaseModelOutputs>() {
                                @Override
                                public void onSuccess(FirebaseModelOutputs result) {
                                    resultView.setText("");
                                    DecimalFormat form = new DecimalFormat("0.00");
                                    float[][] op = result.getOutput(0);
                                    Map<String, Float> o = new HashMap<>();
                                    float max = 0;
                                    String label = "";

//                                    resultView.setText(op.length + "  : " + op.toString());
                                    for(int i = 0; i < op[0].length; i++){
                                        try {
                                            if (Float.parseFloat(form.format(op[0][i])) >= max ){
                                                max = Float.parseFloat(form.format(op[0][i]));
                                                label = ImageUtils.getLabel(getAssets().open("labels.json"), i);

//                                                final String resu = label + ": " + form.format(op[0][i]) + "\n";
//
//                                                o.put(label, op[0][1]);
//
//                                                resultView.append(resu);
//                                                progressBar.dismiss();
                                            }
                                            } catch(IOException e){
                                                e.printStackTrace();
                                            }
                                    }
                                    if(max <= 0.6){
                                        resultView.setText("Flower not found!\nChoose a valid image.");
                                    }
                                    else{
                                        resultView.setText(label);
                                    }
                                    progressBar.dismiss();

                                }
                            })
                    .addOnFailureListener(
                            new OnFailureListener() {
                                @Override
                                public void onFailure(@NonNull Exception e) {
                                    Toast.makeText(getApplicationContext(),  "Couldn't run through model", Toast.LENGTH_LONG).show();
                                }
                            });

        }
        catch (FirebaseMLException e){
            e.printStackTrace();
        }


        //Runs inference in background thread
//        new AsyncTask<Integer,Integer,Integer>(){
//
//            @Override
//
//            protected Integer doInBackground(Integer ...params){
//
//                //Resize the image into 224 x 224
//                Bitmap resized_image = ImageUtils.processBitmap(bitmap,200);
//
//                //Normalize the pixels
//                floatValues = ImageUtils.normalizeBitmap(resized_image,200,127.5f,1.0f);
//
//                //Pass input into the tensorflow
//                tf.feed(INPUT_NAME,floatValues,1,200,200,3);
//
//                //compute predictions
//                tf.run(new String[]{OUTPUT_NAME});
//
//                //copy the output into the PREDICTIONS array
//                tf.fetch(OUTPUT_NAME,PREDICTIONS);
//
//                //Obtained highest prediction
//                Object[] results = argmax(PREDICTIONS);
//
//
//                int class_index = (Integer) results[0];
//                float confidence = (Float) results[1];
//
//
//                try{
//
//                    final String conf = String.valueOf(confidence * 100).substring(0,5);
//
//
//                    DecimalFormat form = new DecimalFormat("0.00");
//                    String res = "";
//
//                    runOnUiThread(new Runnable() {
//                        @Override
//                        public void run() {
//                            resultView.setText("");
//                        }
//                    });
//
//
//                    for(int i = 0; i < 17 ; i++){
//                        final String label = ImageUtils.getLabel(getAssets().open("labels.json"),i);
//                        final String resu =  label + ": " + form.format(PREDICTIONS[i]) + "\n";
//
//                        runOnUiThread(new Runnable() {
//                            @Override
//                            public void run() {
//
//
//                                resultView.append(resu);
//
//                            }
//                        });
//
//                    }
//
//                    progressBar.dismiss();
//
//                    //Convert predicted class index into actual label name
//
//
//
////                    //Display result on UI
////                    runOnUiThread(new Runnable() {
////                        @Override
////                        public void run() {
////
////                            progressBar.dismiss();
//////                            resultView.setText(res);
////
////                        }
////                    });
//
//                }
//
//                catch (Exception e){
//
//
//                }
//
//
//                return 0;
//            }
//
//
//
//        }.execute(0);
//
    }
}
