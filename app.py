# app.py

import os
import io
import base64
import json
import time
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, stream_with_context

# ==============================================================================
# 全局配置和初始化
# ==============================================================================

app = Flask(__name__)

DATA_DIR = r"D:\0-自研工具\ai1\mnist"
MODEL_PATH = 'mnist_model.h5'

# 全局变量
x_train, y_train, x_test, y_test = None, None, None, None
model = None
# 用于存储用户添加的自定义数据 (在内存中，重启后会清空)
custom_data = {'images': [], 'labels': []}
# 挑选一组固定的测试图片用于实时展示预测效果
fixed_test_images = None
fixed_test_labels = None


# ==============================================================================
# 数据加载函数
# ==============================================================================
def load_mnist_data():
    global x_train, y_train, x_test, y_test, fixed_test_images, fixed_test_labels
    print("--- 正在从本地 .npy 文件加载数据... ---")
    x_train_path = os.path.join(DATA_DIR, 'x_train.npy')
    # ... (此处省略，请保留你原来的数据加载代码)
    y_train_path = os.path.join(DATA_DIR, 'y_train.npy')
    x_test_path = os.path.join(DATA_DIR, 'x_test.npy')
    y_test_path = os.path.join(DATA_DIR, 'y_test.npy')
    if not all(map(os.path.exists, [x_train_path, y_train_path, x_test_path, y_test_path])):
        print(f"\n[错误!]：一个或多个 .npy 文件未在指定路径 '{DATA_DIR}' 下找到。")
        return False
    x_train, y_train = np.load(x_train_path), np.load(y_train_path)
    x_test, y_test = np.load(x_test_path), np.load(y_test_path)

    # 随机挑选20张固定的测试图片用于可视化
    indices = np.random.choice(len(x_test), 20, replace=False)
    fixed_test_images = x_test[indices]
    fixed_test_labels = y_test[indices]
    print("--- 本地数据加载完成，并已选取固定测试样本 ---")
    return True


# ==============================================================================
# 模型定义、训练和加载函数 (重大更新)
# ==============================================================================

def create_simple_model():  # ... (此函数不变)
    simple_model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    simple_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return simple_model


def create_cnn_model():  # ... (此函数不变)
    cnn_model = keras.models.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax"),
    ])
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return cnn_model


def send_event(event_name, data):
    """一个辅助函数，用于格式化并发送SSE事件"""
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


class ProgressCallback(keras.callbacks.Callback):
    """自定义回调函数，现在会发送结构化的JSON数据"""

    def __init__(self, yield_func, total_epochs, is_cnn):
        super().__init__()
        self.yield_func = yield_func
        self.total_epochs = total_epochs
        self.is_cnn = is_cnn

    def on_epoch_begin(self, epoch, logs=None):
        # 预先发送一次固定图片的预测结果
        self.update_fixed_predictions(epoch)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # 发送日志事件
        log_data = {
            'epoch': epoch + 1,
            'total_epochs': self.total_epochs,
            'accuracy': logs.get('accuracy', 0),
            'loss': logs.get('loss', 0),
            'val_accuracy': logs.get('val_accuracy', 0),
            'val_loss': logs.get('val_loss', 0),
        }
        self.yield_func(send_event('epoch_log', log_data))
        # 更新固定图片的预测结果
        self.update_fixed_predictions(epoch + 1)

    def update_fixed_predictions(self, epoch):
        images_to_predict = fixed_test_images / 255.0
        if self.is_cnn:
            images_to_predict = np.expand_dims(images_to_predict, -1)

        predictions = self.model.predict(images_to_predict, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        results = []
        for i in range(len(fixed_test_images)):
            img_pil = Image.fromarray(fixed_test_images[i]).convert('L')
            buffer = io.BytesIO()
            img_pil.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append({
                'image_base64': f"data:image/png;base64,{img_str}",
                'true_label': int(fixed_test_labels[i]),
                'predicted_label': int(predicted_labels[i])
            })
        self.yield_func(send_event('prediction_update', {'epoch': epoch, 'predictions': results}))


def train_model_generator(model_type, epochs, use_custom_data):
    """训练生成器，现在会发送丰富的JSON数据"""
    global model, x_train, y_train, x_test, y_test

    yield send_event('log', {'message': "准备数据和模型..."})
    time.sleep(1)

    # 1. 准备数据
    x_train_normalized = x_train / 255.0
    y_train_copy = y_train.copy()  # 使用副本以防修改原始数据

    # 如果使用自定义数据，则进行增强
    if use_custom_data and custom_data['images']:
        yield send_event('log', {'message': f"检测到 {len(custom_data['images'])} 条自定义数据，正在进行数据增强..."})
        custom_images_np = np.array(custom_data['images'])
        custom_labels_np = np.array(custom_data['labels'])
        x_train_normalized = np.concatenate((x_train_normalized, custom_images_np))
        y_train_copy = np.concatenate((y_train_copy, custom_labels_np))
        yield send_event('log', {'message': f"数据增强完成，总训练样本: {len(x_train_normalized)}"})

    x_test_normalized = x_test / 255.0

    is_cnn = model_type == 'cnn'
    if is_cnn:
        x_train_normalized = np.expand_dims(x_train_normalized, -1)
        x_test_normalized = np.expand_dims(x_test_normalized, -1)

    # 2. 创建模型
    yield send_event('log', {'message': f"创建 {'强大的 CNN' if is_cnn else '简单的 Dense'} 模型..."})
    temp_model = create_cnn_model() if is_cnn else create_simple_model()
    time.sleep(1)

    # 3. 开始训练
    yield send_event('log', {'message': "开始训练..."})
    progress_callback = ProgressCallback(lambda log: (yield log), epochs, is_cnn)

    history = temp_model.fit(
        x_train_normalized, y_train_copy,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[progress_callback],
        verbose=0
    )

    yield send_event('log', {'message': "训练完成！正在评估最终模型..."})
    time.sleep(1)

    # 4. 评估模型
    test_loss, test_acc = temp_model.evaluate(x_test_normalized, y_test, verbose=0)
    yield send_event('log', {'message': f"最终测试集准确率: {test_acc:.4f}, 损失: {test_loss:.4f}"})

    # 5. 保存并更新全局模型
    yield send_event('log', {'message': "正在保存新模型..."})
    temp_model.save(MODEL_PATH)
    model = temp_model
    yield send_event('log', {'message': f"新模型已保存到 {MODEL_PATH} 并已激活！"})

    # 发送一个完成信号
    yield send_event('training_complete', {'accuracy': test_acc, 'loss': test_loss})


def load_existing_model():  # ... (此函数不变)
    global model
    if os.path.exists(MODEL_PATH):
        print(f"--- 正在加载已存在的模型 '{MODEL_PATH}' ---")
        model = keras.models.load_model(MODEL_PATH)
        print("--- 模型加载成功 ---")
        return True
    return False


# ==============================================================================
# Flask 路由和视图函数
# ==============================================================================

def initialize_app():
    if not load_mnist_data():
        print("致命错误: 数据加载失败。")
        return
    load_existing_model()


# --- 主页、数据预览、手写识别器的路由和API保持不变 ---
# (此处省略，请保留你原来的代码)
@app.route('/')
def index():
    if x_train is None:
        return "数据加载失败，请检查服务器日志。", 500
    model_info = "未加载 (请先前往“模型训练”页面进行训练)"
    if model:
        is_cnn = any('conv2d' in layer.name for layer in model.layers)
        model_info = f"{'强大的 CNN' if is_cnn else '简单的 Dense'} 模型 (已加载)"
    summary = {
        'x_train_shape': str(x_train.shape), 'y_train_shape': str(y_train.shape),
        'x_test_shape': str(x_test.shape), 'y_test_shape': str(y_test.shape),
        'num_train_images': x_train.shape[0], 'num_test_images': x_test.shape[0],
        'image_dim': f"{x_train.shape[1]}x{x_train.shape[2]}",
        'model_status': model_info,
        'custom_data_count': len(custom_data['images'])
    }
    return render_template('index.html', summary=summary)


@app.route('/data_viewer')
def data_viewer(): return render_template('data_viewer.html')


@app.route('/api/random_images/<int:count>')
def get_random_images(count):
    # ... (此处省略，请保留你原来的代码)
    if x_train is None: return jsonify({"error": "数据尚未加载"}), 500
    images_data = []
    num_to_fetch = min(count, x_train.shape[0])
    for _ in range(num_to_fetch):
        idx = np.random.randint(0, x_train.shape[0])
        image, label = x_train[idx], int(y_train[idx])
        img_pil = Image.fromarray(image).convert('L')
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images_data.append({'label': label, 'image_base64': f"data:image/png;base64,{img_str}"})
    return jsonify(images_data)


@app.route('/handwriting_recognizer')
def handwriting_recognizer(): return render_template('handwriting_recognizer.html')


@app.route('/api/predict', methods=['POST'])
def predict_handwriting():
    # ... (此处省略，请保留你原来的代码)
    if model is None: return jsonify({"error": "模型尚未加载或训练"}), 500
    data = request.get_json()
    img_data_url = data['image']
    _, encoded_data = img_data_url.split(',', 1)
    img_pil = Image.open(io.BytesIO(base64.b64decode(encoded_data)))
    img_processed = img_pil.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
    img_array = 255 - np.array(img_processed)
    img_array_normalized = img_array / 255.0
    if np.max(img_array_normalized) < 0.1: return jsonify(
        {"prediction": "无法识别", "message": "图片内容为空白或太淡，请重试。"}), 200

    is_cnn = any('conv2d' in layer.name for layer in model.layers)
    img_for_prediction = img_array_normalized.reshape(1, 28, 28, 1) if is_cnn else np.expand_dims(img_array_normalized,
                                                                                                  axis=0)

    predictions = model.predict(img_for_prediction, verbose=0)
    predicted_digit = int(np.argmax(predictions[0]))
    probabilities = [f"{p * 100:.2f}%" for p in predictions[0]]
    return jsonify({"prediction": predicted_digit, "probabilities": probabilities, "message": "预测成功"})


# --- 训练路由 ---
@app.route('/train')
def train_page():
    return render_template('train.html', custom_data_count=len(custom_data['images']))


@app.route('/start_training', methods=['GET'])
def start_training():
    model_type = request.args.get('model_type', 'simple')
    epochs = int(request.args.get('epochs', 5))
    use_custom_data = request.args.get('use_custom_data') == 'true'
    return Response(stream_with_context(train_model_generator(model_type, epochs, use_custom_data)),
                    content_type='text/event-stream')


@app.route('/training_complete')
def training_complete():
    accuracy = request.args.get('accuracy', 'N/A')
    loss = request.args.get('loss', 'N/A')
    return render_template('training_complete.html', accuracy=accuracy, loss=loss)


# --- 新增：自定义数据路由 ---
@app.route('/add_data')
def add_data_page():
    # 将已添加的数据传递给模板，以便显示
    b64_images = []
    for img_array in custom_data['images']:
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8)).convert('L')
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        b64_images.append(f"data:image/png;base64,{img_str}")

    return render_template('add_data.html', custom_samples=zip(b64_images, custom_data['labels']))


@app.route('/api/add_custom_data', methods=['POST'])
def add_custom_data():
    data = request.get_json()
    label = int(data['label'])
    img_data_url = data['image']

    # 和手写识别器一样的预处理
    _, encoded_data = img_data_url.split(',', 1)
    img_pil = Image.open(io.BytesIO(base64.b64decode(encoded_data)))
    img_processed = img_pil.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
    img_array = 255 - np.array(img_processed)
    img_array_normalized = img_array / 255.0

    custom_data['images'].append(img_array_normalized)
    custom_data['labels'].append(label)

    return jsonify({"success": True, "message": f"成功添加一张 '{label}' 的图片!", "total": len(custom_data['images'])})


@app.route('/api/clear_custom_data', methods=['POST'])
def clear_custom_data():
    custom_data['images'].clear()
    custom_data['labels'].clear()
    return jsonify({"success": True, "message": "自定义数据已清空。"})


# ==============================================================================
# Flask 应用启动
# ==============================================================================
if __name__ == '__main__':
    print("--- 正在初始化应用... ---")
    initialize_app()
    print("\n--- 启动 Flask Web 应用 ---")
    print("访问地址: http://127.0.0.1:5000")
    print("----------------------------")
    app.run(debug=True, threaded=True)  # threaded=True 对SSE性能有好处