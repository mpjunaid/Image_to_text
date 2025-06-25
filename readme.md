The given data set contained data with the following Images/train, images/test
Groundthruth
example of ground thruth

x: [[115 503 494 115]], y: [[322 346 426 404]], ornt: [u'm'], transcriptions: [u'nauGHTY']
x: [[ 734 1058 1061  744]], y: [[360 369 449 430]], ornt: [u'm'], transcriptions: [u'NURIS']
x: [[558 682 682 557]], y: [[370 375 404 398]], ornt: [u'm'], transcriptions: [u'NURIS']
x: [[562 595 651 687 653 637 604 588]], y: [[347 304 305 360 366 334 332 361]], ornt: [u'c'], transcriptions: [u'nauGHTY']
x: [[603 632 630 603]], y: [[408 413 426 423]], ornt: [u'h'], transcriptions: [u'EST']
x: [[599 638 637 596]], y: [[419 422 441 437]], ornt: [u'h'], transcriptions: [u'1996']
x: [[583 602 633 656 679 648 594 558]], y: [[410 445 445 411 428 476 472 432]], ornt: [u'c'], transcriptions: [u'warunG']
x: [[543 583 660 701 691 653 592 557]], y: [[347 288 288 347 358 308 302 355]], ornt: [u'#'], transcriptions: [u'#']
x: [[557 580 640 683 698 649 583 537]], y: [[419 470 481 422 432 497 491 432]], ornt: [u'#'], transcriptions: [u'#']

Using this data I have wrote a python code called preprocess.py which contains the code to dwnload and extract both the images and annptations and then cropped the images based on the given transctipons and image cordinaeted and created a new data set called cropped

After that 

First experiment was to run the model on a preexisiting OCR model to see the performance and got the results and saved it in cropped/predictions.csv and using [cal_](cal_precision_TrOCR_direct.py)
got a result of precision to be Exact Match Precision: 0.1386 (306/2208)

Further attempts where made to Fine tune the same model but coudnt continue due not having enough compute power and dropped continuing forward as the target was to develop and train a custome model.

then the first very basic model with the following architecture was created
Absolutely! Here's the cleaned and formatted content in Markdown (.md) format, ready for copy-paste:

# OCR Dataset Processing and Model Training

## Dataset Description

The provided dataset consists of image data organized as follows:

images/
├── train/
├── test/
Groundtruth/


### Example Ground Truth Format

Each ground truth annotation contains:
- `x`: X-coordinates of the bounding box
- `y`: Y-coordinates of the bounding box
- `ornt`: Orientation of the text
- `transcriptions`: Actual text label

Example:

x: [[115 503 494 115]], y: [[322 346 426 404]], ornt: ['m'], transcription: ['nauGHTY']
x: [[734 1058 1061 744]], y: [[360 369 449 430]], ornt: ['m'], transcription: ['NURIS']
x: [[558 682 682 557]], y: [[370 375 404 398]], ornt: ['m'], transcription: ['NURIS']
x: [[562 595 651 687 653 637 604 588]], y: [[347 304 305 360 366 334 332 361]], ornt: ['c'], transcription: ['nauGHTY']
x: [[603 632 630 603]], y: [[408 413 426 423]], ornt: ['h'], transcription: ['EST']
x: [[599 638 637 596]], y: [[419 422 441 437]], ornt: ['h'], transcription: ['1996']
x: [[583 602 633 656 679 648 594 558]], y: [[410 445 445 411 428 476 472 432]], ornt: ['c'], transcription: ['warunG']
x: [[543 583 660 701 691 653 592 557]], y: [[347 288 288 347 358 308 302 355]], ornt: ['#'], transcription: ['#']
x: [[557 580 640 683 698 649 583 537]], y: [[419 470 481 422 432 497 491 432]], ornt: ['#'], transcription: ['#']


## Preprocessing

The script `preprocess.py` was created to:

1. Download and extract the image and annotation files.
2. Crop the images based on transcription text and bounding box coordinates.
3. Save the cropped images into a new dataset folder called `cropped`.

## Experiments

### 1. Direct Evaluation with Pre-trained OCR

The first experiment involved testing a pre-trained OCR model (e.g., TrOCR) on the cropped dataset.

- Predictions were saved to: `cropped/predictions.csv`
- Precision was calculated using the script: [`cal_precision_TrOCR_direct.py`](cal_precision_TrOCR_direct.py)

**Result:**

Exact Match Precision: 0.1386 (306 / 2208)


### 2. Attempted Fine-tuning

An attempt was made to fine-tune the same pre-trained model. However, due to limited computational resources, this approach was discontinued. The final goal shifted to developing a custom OCR model.

### 3. Custom OCR Model (Baseline)

A basic model was implemented with a simple architecture. The training and test performance are as follows:

#### Results

| Epoch | Training Precision | Test Precision |
|-------|--------------------|----------------|
| 5     | 0.9250             | 0.9450         |
| 100   | 1.0000             | 0.9475         |

- Model outputs and evaluation logs are saved in the `result_base_model/` directory.
- Loss and accuracy graphs are saved as:
  - `result_base_model/image1.png`
  - `result_base_model/image2.png`

## Flask Web App

A simple Flask application was developed for demonstration purposes.

- Code is located in the `flask/` directory.
- Example screenshots:
  - `flask/screenshot-1.png`
  - `flask/screenshot-2.png`

Let me know if you'd like image previews (using ![Alt Text](path)) or automatic table of contents added.
You said:
give me just one file of markup text to make it as readme in github
ChatGPT said:

Sure! Here's the full Markdown text for your GitHub README.md file — just copy and save it as README.md in your project root directory:

# OCR Dataset Processing and Model Training