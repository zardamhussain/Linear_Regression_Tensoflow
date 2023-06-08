const X = [], Y = [];
let m, b; 

const learningRate = 0.4;
const optimizer = tf.train.sgd(learningRate);


function setup(){
    createCanvas(windowWidth, windowHeight);
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}


function mousePressed() {
    const x = map(mouseX, 0, width, 0, 1);
    const y = map(mouseY, 0, height, 0, 1);
    X.push(x);
    Y.push(y);
};



function predict(xs) {
    return m.mul(xs).add(b);
}

function loss(predicts, labels) {
    return predicts.sub(labels).square().mean();
}


function drawLine () {
    let x1 = 0;
    let y1 = m.dataSync()[0] * x1 + b.dataSync()[0];
    let x2 = 1;
    let y2 = m.dataSync()[0] * x2 + b.dataSync()[0];
  
    x1 = map(x1, 0, 1, 0, width);
    y1 = map(y1, 0, 1, 0, height);
    x2 = map(x2, 0, 1, 0, width);
    y2 = map(y2, 0, 1, 0, height);
  
    line(x1, y1, x2, y2);
  }


function draw() {
    background(0);
    textFont('monospace');
    let instruction = "Tap / Click on the Screen to Insert Data Points...";
    text(instruction, 15, 30);

    stroke(255);
    strokeWeight(4);
    for(let i=0; i<X.length; ++i) {
        const x = map(X[i], 0, 1, 0, width);
        const y = map(Y[i], 0, 1, 0, height);
        point(x, y);
    }


    if (X.length) {
        tf.tidy(() => {
          const xs = tf.tensor1d(X);
          const ys = tf.tensor1d(Y);
    
          optimizer.minimize(() => loss(predict(xs), ys));
    
        });

        drawLine();
      }
    
}