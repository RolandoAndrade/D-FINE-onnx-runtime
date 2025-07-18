import * as ort from 'onnxruntime-node';
import sharp from 'sharp';

const COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
];


/**
 * Resizes an image while maintaining aspect ratio and pads it.
 * @param {sharp.Sharp} image - The sharp image instance.
 * @param {number} size - The target size (width and height).
 * @returns {Promise<{buffer: Buffer, ratio: number, padW: number, padH: number, originalWidth: number, originalHeight: number}>}
 */
async function resizeWithAspectRatio(image, size) {
    const metadata = await image.metadata();
    const originalWidth = metadata.width;
    const originalHeight = metadata.height;

    const ratio = Math.min(size / originalWidth, size / originalHeight);
    const newWidth = Math.round(originalWidth * ratio);
    const newHeight = Math.round(originalHeight * ratio);

    const resizedImage = await image.resize(size, size, {
        fit: 'contain',
        background: { r: 114, g: 114, b: 114 }, // Use a neutral gray for padding
    });

    // Padding is calculated by sharp's 'contain' fit, but we can get it for bbox conversion
    const padW = (size - newWidth) / 2;
    const padH = (size - newHeight) / 2;

    const paddedImageBuffer = await resizedImage.raw().toBuffer();

    return { buffer: paddedImageBuffer, ratio, padW, padH, originalWidth, originalHeight };
}

/**
 * Draws bounding boxes and labels on an image.
 * @param {string | Buffer} imagePathOrBuffer - Path to the image or image buffer.
 * @param {Array<{label: string, box: number[], score: number}>} detections - The array of detected objects.
 * @returns {Promise<sharp.Sharp>}
 */
async function draw(imagePathOrBuffer, detections) {
    const image = sharp(imagePathOrBuffer);
    const metadata = await image.metadata();
    const { width: originalWidth, height: originalHeight } = metadata;

    const svgElements = [];
    for (const detection of detections) {
        const { label, box, score } = detection;
        const [x1, y1, x2, y2] = box;

        console.log(`Detected: ${label} at [${x1.toFixed(0)}, ${y1.toFixed(0)}, ${x2.toFixed(0)}, ${y2.toFixed(0)}] with score ${score.toFixed(2)}`);

        const rect = `<rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" style="stroke:red; stroke-width:3; fill-opacity:0"/>`;
        const text = `<text x="${x1}" y="${y1 - 5}" style="font-size:16px; fill:blue; font-weight: bold;">${label} (${score.toFixed(2)})</text>`;
        svgElements.push(rect, text);
    }

    if (svgElements.length === 0) {
        console.log("No objects detected above the threshold.");
        return image;
    }

    const svgOverlay = `<svg width="${originalWidth}" height="${originalHeight}">${svgElements.join('')}</svg>`;
    return image.composite([{ input: Buffer.from(svgOverlay), blend: 'over' }]);
}

/**
 * Converts an image buffer to a tensor.
 * @param {Buffer} buffer - The raw image buffer (RGB).
 * @param {number} width - The image width.
 * @param {number} height - The image height.
 * @returns {ort.Tensor}
 */
function imageToTensor(buffer, width, height) {
    const float32Data = new Float32Array(width * height * 3);
    for (let i = 0; i < buffer.length; i += 3) {
        const pixelIndex = i / 3;
        const r = buffer[i] / 255.0;
        const g = buffer[i + 1] / 255.0;
        const b = buffer[i + 2] / 255.0;
        // HWC to CHW
        float32Data[pixelIndex] = r;
        float32Data[width * height + pixelIndex] = g;
        float32Data[width * height * 2 + pixelIndex] = b;
    }
    return new ort.Tensor('float32', float32Data, [1, 3, height, width]);
}

/**
 * Processes a single image.
 * @param {ort.InferenceSession} sess - The ONNX session.
 * @param {string} imagePath - The path to the image.
 */
async function processImage(sess, imagePath) {
    const image = sharp(imagePath);
    const { buffer, ratio, padW, padH, originalWidth, originalHeight } = await resizeWithAspectRatio(image, 640);
    const imageTensor = imageToTensor(buffer, 640, 640);

    const feeds = {
        pixel_values: imageTensor,
    };

    const output = await sess.run(feeds);
    const { logits, pred_boxes: boxes } = output;

    // Post-process logits to get scores and labels
    const logitsData = logits.data;
    const boxesData = boxes.data;
    const numDetections = logits.dims[1];
    const numClasses = logits.dims[2];


    const noObjectClassIndex = numClasses - 1;

    const detections = [];
    for (let i = 0; i < numDetections; i++) {
        const classScores = logitsData.subarray(i * numClasses, (i + 1) * numClasses);

        let maxScore = -Infinity;
        let maxIndex = -1;
        for (let j = 0; j < classScores.length; j++) {
            if (classScores[j] > maxScore) {
                maxScore = classScores[j];
                maxIndex = j;
            }
        }

        // Skip if the most likely class is 'no object'
        if (maxIndex === noObjectClassIndex) {
            continue;
        }

        // Apply softmax to get a score between 0 and 1
        const expSum = classScores.reduce((sum, score) => sum + Math.exp(score - maxScore), 0);
        const score = 1 / expSum; // Simplified softmax for the max score

        const confidenceThreshold = 0.4; // Use a reasonable threshold
        if (score > confidenceThreshold) {
            const label = COCO_LABELS[maxIndex] || `Class ${maxIndex}`;

            const boxIndex = i * 4;
            const [cx, cy, w, h] = [
                boxesData[boxIndex],
                boxesData[boxIndex + 1],
                boxesData[boxIndex + 2],
                boxesData[boxIndex + 3],
            ];

            // Convert from [center_x, center_y, w, h] (normalized) to [x1, y1, x2, y2] (absolute)
            const abs_x1 = ((cx - w / 2) * 640 - padW) / ratio;
            const abs_y1 = ((cy - h / 2) * 640 - padH) / ratio;
            const abs_x2 = ((cx + w / 2) * 640 - padW) / ratio;
            const abs_y2 = ((cy + h / 2) * 640 - padH) / ratio;

            detections.push({ label, score, box: [abs_x1, abs_y1, abs_x2, abs_y2] });
        }
    }

    const resultImage = await draw(imagePath, detections);
    await resultImage.toFile('onnx_result.jpg');
    console.log("Image processing complete. Result saved as 'onnx_result.jpg'.");
}

const image = process.argv[2] || '20241130_102008.jpg'

/**
 * Main function.
 */
async function main() {
    try {
        const sess = await ort.InferenceSession.create('./model_lg.onnx');
        const inputNames = sess.inputNames;
        console.log('Input names:', inputNames);
        console.log(`Using device: ${ort.env.wasm.simd ? 'SIMD' : 'default'}`);
        await processImage(sess, image);
    } catch (e) {
        console.error('An error occurred:', e);
    }
}

main();