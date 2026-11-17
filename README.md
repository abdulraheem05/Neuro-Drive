<div align="center">

<img src="./docs/images/neurodrive-logo.png" alt="NeuroDrive Logo" width="800" align="center">

<br>
<br>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep_Learning-D00000.svg?logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Data_Processing-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![CNN](https://img.shields.io/badge/CNN-Behavioral_Cloning-8A2BE2.svg)]()
[![Udacity](https://img.shields.io/badge/Udacity-Driving_Simulator-02B3E4.svg?logo=udacity&logoColor=white)](https://github.com/udacity/self-driving-car-sim)
[![Autonomous Driving](https://img.shields.io/badge/Autonomous-Driving-00A86B.svg)]()

<br>
</div>

NeuroDrive is a computer-vision project that learns driving behaviour from human demonstrations. A CNN takes a front-facing camera image and predicts **steering** and **throttle** commands.

**Selected model:** V8 · **Demonstration:** Udacity Simulator, Location 01

<img src="./docs/images/udacity page.png" alt="NeuroDrive Logo" width="800" align="center">

## How it works

```mermaid
flowchart TD
 A[Human drives in simulator] --> B[Camera images + driving log]
 B --> C[Dataset inspection and preprocessing]
 C --> D[Train / validation / test split]
 D --> E[CNN training]
 E --> F[Selected Keras model]
 G[Simulator sends image to drive.py] --> H[RGB crop and resize]
 F --> H
 H --> I[Predict steering + throttle]
 I --> J[Send controls to simulator]
```

`drive.py` runs a local Socket.IO server on port `4567`. It decodes each Base64 camera image, preprocesses it, predicts both controls and sends them back to the simulator.

## Data collection

Data was recorded by manually driving in the Udacity simulator. The simulator saves camera images and a `driving_log.csv` containing the corresponding image paths and control values.

Approximately **15,000 images** were collected across multiple attempts. Different driving patterns and collection rounds were needed because the early model versions did not achieve the desired driving behaviour.

### Three camera views

![Centre, left and right camera views](./docs/images/three_camera_views.png)

| CSV field | Description |
|---|---|
| `center`, `left`, `right` | Three camera image paths |
| `steering` | Steering command |
| `throttle` | Accelerator command |
| `brake` | Brake value |
| `speed` | Vehicle speed |

Training randomly selects centre, left or right images. Side-camera samples receive a steering correction (`+0.20` for left, `-0.20` for right). Validation and testing use centre-camera images.

## Preprocessing

The dataset is checked for missing values and missing image files, then split using seed `42`:

| Split | Data |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Test | 20% |

Images are loaded with OpenCV (BGR), cropped by removing the top 35% and bottom 10%, resized to **200 × 66**, and converted to **RGB**. The resulting model input is `66 × 200 × 3`.

Training augmentation includes random camera selection, steering correction, horizontal flipping (with steering sign reversal) and brightness adjustment. Validation and test preprocessing is deterministic.

![Preprocessing example](./docs/images/preprocessing_example.png)

## CNN architecture

The network is inspired by NVIDIA's end-to-end driving CNN. It uses a shared convolutional feature extractor and two regression outputs.

![Architecture reference](./docs/images/cnn_architecture_notebook.png)

## Training and experiments

V8 training configuration:

| Setting | Value |
|---|---|
| Framework / optimiser | TensorFlow / Adam |
| Learning rate | `0.0001` |
| Loss | MSE for both outputs |
| Batch size | `64` |
| Maximum epochs | `40` |
| Random seed | `42` |

ModelCheckpoint, EarlyStopping, ReduceLROnPlateau and CSVLogger are used during training. The saved V8 history contains **27 epochs**.

![Callbacks](./docs/images/callbacks.png)

## Mean Absolute Error (MAE)

MAE measures the average absolute difference between the model's predicted controls and the actual recorded driving values.

### Steering MAE

![V8 Steering MAE](./docs/images/steering_mae_v8.png)

### Throttle MAE

![V8 Throttle MAE](./docs/images/throttle_mae_v8.png)

Lower MAE indicates predictions closer to the recorded driving behaviour.

## Run NeuroDrive Locally

### 1. Download the Udacity Simulator

Download the **Udacity Self-Driving Car Simulator (Beta)** for your operating system:

- [Windows / Linux / macOS – Simulator Releases](https://github.com/udacity/self-driving-car-sim)

Extract the downloaded archive and launch the simulator.

### 2. Clone the Repository

```bash
git clone https://github.com/abdulraheem05/Neuro-Drive.git
cd NeuroDrive
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Launch Autonomous Driving

1. Open the Udacity Simulator.
2. Select **Location 01**.
3. Click **Autonomous Mode**.
4. Open a terminal in the NeuroDrive project directory.
5. Run:

```bash
python drive.py
```

The Python server loads the trained model and communicates with the simulator through Socket.IO on port `4567`. The vehicle should now drive autonomously using predicted steering and throttle commands.

> **Note:** Ensure the trained model is available in the `models/` directory before running `drive.py`.

### Reproducing Training

The raw driving dataset is not included. To train the model from scratch, collect your own simulator data using the Training Mode in Udacity and follow the notebooks provided in `notebooks/`.

## Limitations

- Demonstrated on simulator Location 01; other locations may behave differently.
- Performance depends on the driving patterns represented in the training data.
- Throttle is learned from demonstrations; it is not a closed-loop speed controller.
- Not tested or intended for real-world vehicle control.

## Acknowledgements

- [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)
- [NVIDIA end-to-end learning for autonomous driving](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
