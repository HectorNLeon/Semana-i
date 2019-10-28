let net;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
const counter = document.getElementById('counter');
let count = [0,0,0,0,0];

async function setupWebcam() {
    return new Promise((resolve, reject) => {
      const navigatorAny = navigator;
      navigator.getUserMedia = navigator.getUserMedia ||
          navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
          navigatorAny.msGetUserMedia;
      if (navigator.getUserMedia) {
        navigator.getUserMedia({video: true},
          stream => {
            console.log(webcamElement)
            webcamElement.srcObject = stream;
            webcamElement.addEventListener('loadeddata',  () => resolve(), false);
          },
          error => reject());
      } else {
        reject();
      }
    });
}

function knnLoad(){
  //can be change to other source
  let tensorObj = JSON.parse(window.localStorage.getItem('knnClassifier'));
  //covert back to tensor
  Object.keys(tensorObj).forEach((key) => {
    tensorObj[key] = tf.tensor(tensorObj[key], [Math.floor(tensorObj[key].length / 1000), 1024]);
    console.log(Math.floor(tensorObj[key].length));
  });
  // tu codigo va a aquí
  console.log(tensorObj);
  classifier.setClassifierDataset(tensorObj);
  
  ///
}
  

async function app(){
    console.log('Loading mobilenet..');

    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model')
    
    knnLoad();
  
    await setupWebcam();

    //Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = async classId => {
        count[classId]++;
        // Capture an image from the web camera.
        let className = ['Coca','Cafe','Coca light','Sabritas','Emperador'];
        counter.innerText = `
              class: ${className[classId]}\n
              ${count[classId]}
        `;

        // Get the intermediate activation of MobileNet 'conv_preds' and pass that
        // to the KNN classifier.
        const activation = net.infer(webcamElement, 'conv_preds');

        // Pass the intermediate activation to the classifier.
        classifier.addExample(activation, classId);

        // Dispose the tensor to release the memory.
    };
    // When clicking a button, add an example for that class.
    /*
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));
    document.getElementById('class-d').addEventListener('click', () => addExample(3));
    document.getElementById('class-e').addEventListener('click', () => addExample(4));
    document.getElementById('save').addEventListener('click', () => save());
*/



    while (true) {
        if (classifier.getNumClasses() > 0) {      
            // Get the activation from mobilenet from the webcam.
            const activation = net.infer(webcamElement, 'conv_preds');
            // Get the most likely class and confidences from the classifier module.
            const result = await classifier.predictClass(activation);
            
            const classes = ['Coca','Cafe','Coca light','Sabritas','Emperador'];
            document.getElementById('console').innerText = `
              prediction: ${classes[result.label]}\n
              probability: ${result.confidences[result.label]}
            `;
      
            // Dispose the tensor to release the memory.
          }      

        // Give some breathing room by waiting for the next animation frame to
        // fire.
        await tf.nextFrame();
    }
}

function saveData(content, name){
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(content);
    var downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href",     dataStr);
    downloadAnchorNode.setAttribute("download", name);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

function save() {
	//Aqui va tu codigo
    let dataset = classifier.getClassifierDataset();
    console.log(dataset);
	//
   var datasetObj = {}
   Object.keys(dataset).forEach((key) => {
     let data = dataset[key].dataSync();
     datasetObj[key] = Array.from(data);
   });
   let jsonStr = JSON.stringify(datasetObj);
   localStorage.setItem("knnClassifier", jsonStr);
   saveData(jsonStr, 'knnClassifierAndatti.json');
 }


app();