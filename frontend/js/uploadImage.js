function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];

    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
    })
        .then(response => response.json())
        .then(data => handleResponse(data, file))
        .catch(handleError);
}

// 1. Переключение секций
function toggleSections() {
    document.getElementById('section_1').classList.add('hidden');
    document.getElementById('section_2').classList.remove('hidden');
    document.getElementById('resultsContainer').classList.remove('hidden');
}

// 2. Отображение изображений
function displayImages(file, processedImageBase64) {
    document.getElementById('originalImage').src = URL.createObjectURL(file);

    if (processedImageBase64) {
        document.getElementById('processedImage').src = `data:image/png;base64,${processedImageBase64}`;
    }
}

// 3. Заполнение таблицы значениями и решениями анализа
function fillResults(data) {
    document.getElementById('meanMagnitude').textContent = data.mean_magnitude;
    document.getElementById('stdMagnitude').textContent = data.std_magnitude;
    document.getElementById('freqEnergyRatio').textContent = data.freq_energy_ratio;
    document.getElementById('angularSymmetry').textContent = data.angular_symmetry;
    document.getElementById('freqEntropy').textContent = data.frequency_entropy;
    document.getElementById('periodicityRatio').textContent = data.periodicity_ratio;
    document.getElementById('laplacianVariance').textContent = data.laplacian_variance;
    document.getElementById('dftAnalysisDecision').textContent = data.dft_Analysis_Decision;
    document.getElementById('blurAnalysisDecision').textContent = data.blur_Analysis_Decision;
    document.getElementById('overallDecision').textContent = data.overall_decision;
}

// 4. Обработка ответа
function handleResponse(data, file) {
    toggleSections();
    displayImages(file, data.mag_spectrum_base64);
    fillResults(data);
}

// 5. Обработка ошибок
function handleError(error) {
    alert('Error uploading image.');
    console.error('Error:', error);
}
