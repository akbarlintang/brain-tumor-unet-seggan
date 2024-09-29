from django.shortcuts import render

# ML library
import nibabel as nib
import numpy as np
import cv2
from io import BytesIO
import io
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import base64
import tempfile

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Create your views here.
def homepage(request):
    if request.method == 'POST' and 'flairFile' in request.FILES and 'ceFile' in request.FILES:
        # Get the uploaded files directly from the request
        flair_file = request.FILES['flairFile']
        ce_file = request.FILES['ceFile']

        # Use temporary files to save and load the NIfTI files
        with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp_flair_file, \
             tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp_ce_file:
            # Write the BytesIO data to temporary files
            tmp_flair_file.write(flair_file.read())
            tmp_ce_file.write(ce_file.read())

            # Get the temporary file paths
            tmp_flair_path = tmp_flair_file.name
            tmp_ce_path = tmp_ce_file.name

        try:
            # Load the NIfTI images from the temporary files
            flair_img = nib.load(tmp_flair_path).get_fdata()[:, :, 60]  # Example slice index 60
            ce_img = nib.load(tmp_ce_path).get_fdata()[:, :, 60]  # Example slice index 60

            # Perform segmentation prediction
            prediction_image_base64 = showPredictForSingleImage(flair_img, ce_img)

        except nib.filebasedimages.ImageFileError as e:
            return render(request, 'home.html', {
                'error': f'Error loading NIfTI files: {str(e)}'
            })
        except Exception as e:
            return render(request, 'home.html', {
                'error': f'An unexpected error occurred: {str(e)}'
            })
        finally:
            # Clean up temporary files
            os.remove(tmp_flair_path)
            os.remove(tmp_ce_path)

        # Render the template after processing
        return render(request, 'home.html', {
            'prediction_image_base64': prediction_image_base64
        })
    
    return render(request, 'home.html')

# DEFINE seg-areas
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'EDEMA',
    2 : 'NECROTIC/CORE',
    3 : 'ENHANCING'
}

VOLUME_SLICES = 100
IMG_SIZE=128

model_path = os.path.join(os.path.dirname(__file__), 'models/generator_model.h5')
generator_model = tf.keras.models.load_model(model_path)

# ML functions
def predictSingleImage(flair_img, ce_img):
    X = np.empty((1, IMG_SIZE, IMG_SIZE, 2))  # Single image, so shape is (1, IMG_SIZE, IMG_SIZE, 2)

    # Resize the input images to the desired size
    X[0,:,:,0] = cv2.resize(flair_img, (IMG_SIZE, IMG_SIZE))
    X[0,:,:,1] = cv2.resize(ce_img, (IMG_SIZE, IMG_SIZE))

    # Normalize the input
    X = X / np.max(X)

    # Predict the output segmentation
    return generator_model.predict(X, verbose=1)

def showPredictForSingleImage(flair_img, ce_img):
    p = predictSingleImage(flair_img, ce_img)  # Get the predictions

    # Extract the predicted segments
    edema = p[0, :, :, 2]  # Assuming edema is the second index
    core = p[0, :, :, 1]   # Assuming core is the first index
    enhancing = p[0, :, :, 3]  # Assuming enhancing is the third index

    # Resize for overlaying
    flair_resized = cv2.resize(flair_img, (IMG_SIZE, IMG_SIZE))

    # Create a figure with 4 subplots (one for each type)
    plt.figure(figsize=(20, 5))

    # Show the original flair image
    plt.subplot(1, 4, 1)
    plt.imshow(flair_resized, cmap="gray")
    plt.title('Original Image (FLAIR)')
    plt.axis('off')

    # Overlay Edema on the original image
    plt.subplot(1, 4, 2)
    plt.imshow(flair_resized, cmap="gray")
    plt.imshow(edema, cmap="OrRd", alpha=0.5)  # Overlay edema
    plt.title(f'{SEGMENT_CLASSES[1]} Predicted')
    plt.axis('off')

    # Overlay Core on the original image
    plt.subplot(1, 4, 3)
    plt.imshow(flair_resized, cmap="gray")
    plt.imshow(core, cmap="OrRd", alpha=0.5)  # Overlay core
    plt.title(f'{SEGMENT_CLASSES[2]} Predicted')
    plt.axis('off')

    # Overlay Enhancing on the original image
    plt.subplot(1, 4, 4)
    plt.imshow(flair_resized, cmap="gray")
    plt.imshow(enhancing, cmap="OrRd", alpha=0.5)  # Overlay enhancing
    plt.title(f'{SEGMENT_CLASSES[3]} Predicted')
    plt.axis('off')

    # Save the plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the image to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return image_base64