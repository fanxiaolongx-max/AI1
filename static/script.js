// static/script.js

document.addEventListener('DOMContentLoaded', () => {
    // --- 处理数据预览页面 (代码不变) ---
    const dataGallery = document.getElementById('data-gallery');
    if (dataGallery) {
        const loadMoreBtn = document.getElementById('load-more-images');
        loadMoreBtn.addEventListener('click', () => loadRandomImages(9));
        loadRandomImages(9);
    }
    async function loadRandomImages(count) { /* ... 此函数保持不变 ... */
        try {
            const loadMoreBtn = document.getElementById('load-more-images');
            loadMoreBtn.textContent = '加载中...';
            loadMoreBtn.disabled = true;
            const response = await fetch(`/api/random_images/${count}`);
            const images = await response.json();
            images.forEach(item => {
                const imgItem = document.createElement('div');
                imgItem.classList.add('image-item');
                imgItem.innerHTML = `<img src="${item.image_base64}" alt="MNIST Digit"><p>真实数字: ${item.label}</p>`;
                dataGallery.appendChild(imgItem);
            });
            loadMoreBtn.textContent = '加载更多图片';
            loadMoreBtn.disabled = false;
        } catch (error) { console.error('加载图片失败:', error); }
    }

    // --- 处理手写识别器页面 (代码不变) ---
    const canvas = document.getElementById('drawingCanvas');
    if (canvas) { /* ... 此部分代码保持不变 ... */
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.strokeStyle = '#FFF';
        const draw = (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left, y = e.clientY - rect.top;
            ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y);
        };
        canvas.addEventListener('mousedown', (e) => { isDrawing = true; const rect = canvas.getBoundingClientRect(); ctx.beginPath(); ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top); });
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', () => { isDrawing = false; ctx.beginPath(); });
        canvas.addEventListener('mouseout', () => { isDrawing = false; ctx.beginPath(); });
        document.getElementById('clearCanvas').addEventListener('click', () => { ctx.fillRect(0, 0, canvas.width, canvas.height); document.getElementById('prediction-digit').textContent = ''; document.getElementById('probabilities-list').innerHTML = ''; document.getElementById('prediction-message').textContent = ''; });
        document.getElementById('predictDigit').addEventListener('click', async () => {
            const imageDataURL = canvas.toDataURL('image/png');
            document.getElementById('loadingSpinner').style.display = 'block';
            document.getElementById('prediction-message').textContent = '正在识别...';
            try {
                const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: imageDataURL }) });
                const result = await response.json();
                document.getElementById('prediction-digit').textContent = result.prediction !== "无法识别" ? result.prediction : '🤷‍♂️';
                document.getElementById('prediction-message').textContent = result.message;
                const probList = document.getElementById('probabilities-list');
                probList.innerHTML = '';
                result.probabilities.forEach((prob, index) => { const li = document.createElement('li'); li.textContent = `数字 ${index}: ${prob}`; probList.appendChild(li); });
            } catch (error) { document.getElementById('prediction-digit').textContent = '❌'; document.getElementById('prediction-message').textContent = '网络请求失败。'; }
            finally { document.getElementById('loadingSpinner').style.display = 'none'; }
        });
    }

    // --- 新增：处理模型训练页面 ---
    const trainingForm = document.getElementById('training-form');
    if (trainingForm) {
        trainingForm.addEventListener('submit', (e) => {
            e.preventDefault(); // 阻止表单默认提交

            const logContainer = document.getElementById('log-container');
            const logOutput = document.getElementById('log-output');
            const trainBtn = document.getElementById('start-training-btn');

            // 准备UI
            logContainer.style.display = 'block';
            logOutput.textContent = '准备连接到服务器...';
            trainBtn.disabled = true;
            trainBtn.textContent = '正在训练中...';

            const formData = new FormData(trainingForm);

            // 使用 EventSource (Server-Sent Events) 来接收流式数据
            const eventSource = new EventSource(`/start_training?model_type=${formData.get('model_type')}&epochs=${formData.get('epochs')}`);

            let logInitialized = false;

            eventSource.onmessage = function(event) {
                if (!logInitialized) {
                    logOutput.textContent = ''; // 清空初始消息
                    logInitialized = true;
                }
                logOutput.textContent += event.data + '\n';
                // 自动滚动到底部
                logOutput.scrollTop = logOutput.scrollHeight;
            };

            // 监听自定义的 'training_complete' 事件
            eventSource.addEventListener('training_complete', function(event) {
                const data = JSON.parse(event.data);
                logOutput.textContent += '\n🎉 训练全部完成! 正在跳转到结果页面...';

                // 关闭连接
                eventSource.close();

                // 恢复按钮状态
                trainBtn.disabled = false;
                trainBtn.textContent = '开始训练';

                // 跳转到结果页面
                window.location.href = `/training_complete?accuracy=${data.accuracy}&loss=${data.loss}`;
            });

            eventSource.onerror = function(err) {
                logOutput.textContent += '\n❌ 与服务器的连接发生错误。训练已中断。';
                console.error("EventSource failed:", err);
                eventSource.close();
                trainBtn.disabled = false;
                trainBtn.textContent = '开始训练';
            };
        });
    }
});
