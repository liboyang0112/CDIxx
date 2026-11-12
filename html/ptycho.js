// ==================== Page-Specific Initialization ====================
const ptychoImages = [
    'recon_pupil_proj',
    'recon_pupil',
    'recon_pupil_b4_proj',
    'ptycho_afterIterwave',
    'ptycho_probe_afterIter'
];

ptychoImages.forEach(id => {
    selectListenerWithInspect(id)
});

if (document.getElementById('recon_pupil_proj')) {
    document.getElementById('recon_pupil_proj').classList.add('selected');
    operateImage = 'recon_pupil_proj';
    if (document.getElementById('active-image-name')) {
        document.getElementById('active-image-name').textContent = imageNames['recon_pupil_proj'];
    }
}

// ==================== Load Images ====================
function loadImage() {
    const outputName = document.getElementById('ptychoOutputName').value?.trim();
    if (!outputName) {
        showNotification('Please enter output directory name', 'error');
        return;
    }

    setCookie('ptycho_outputname', outputName);
    const timestamp = getTimestamp();
    const path = repo + workpath + "/" + outputName + "/";

    ptychoImages.forEach(imgId => {
        const img = document.getElementById(imgId);
        if (img) {
            img.src = path + imgId + ".jpg?t=" + timestamp;
        }
    });
    displayDiagnosticImages(outputName);
    showNotification('Images loaded from output directory', 'success');
}

// ==================== Event Listeners ====================
document.getElementById("reload")?.addEventListener('click', async () => {
    loadImage();
});

document.getElementById("setpath")?.addEventListener('click', async () => {
    const pathValue = document.getElementById('path').value?.trim();
    if (!pathValue) {
        showNotification('Please enter a working folder path', 'error');
        return;
    }

    workpath = pathValue;
    setCookie('ptychoworkpath', workpath);

    setPath(pathValue, {
        'ptycho.cfg': 'ptychoConfig',
        'ExperimentLog.txt': 'note'
    });

    showNotification('Working folder confirmed: ' + workpath, 'success');

    await listFiles("cd " + repo + workpath + " ; ls -d */ 2>/dev/null | sed 's/\\\\///g'")
        .then(data => {
            autoFillDictionary["ptycho_dataset"] = data;
        });
});

// ==================== GPU Selection ====================
document.getElementById("selGPU_btn")?.addEventListener('click', async () => {
    const cmd = "echo export CUDA_VISIBLE_DEVICES=" + document.getElementById('iGPU').value + " > selectGPU.sh";
    try {
        await fetch(url_cmd, { method: 'POST', body: cmd });
        showNotification('GPU selection saved', 'success');
    } catch (error) {
        showNotification('Error setting GPU: ' + error, 'error');
    }
});

// ==================== Dataset Creation ====================
document.getElementById("createDatasetBtn")?.addEventListener('click', async () => {
    const datasetName = document.getElementById('datasetNameInput').value;
    if (!datasetName) {
        showNotification('Please enter a dataset name', 'error');
        return;
    }

    const path = repo + workpath + "/" + datasetName;
    const cmd = "mkdir -p " + path;

    try {
        await fetch(url_cmd, { method: 'POST', body: cmd });
        document.getElementById('datasetName').value = datasetName;
        showNotification(`Dataset folder created: ${datasetName}`, 'success');

        await listFiles("cd " + repo + workpath + " ; ls -d */ 2>/dev/null | sed 's/\\\\///g'")
            .then(data => {
                autoFillDictionary["ptycho_dataset"] = data;
            });
    } catch (error) {
        showNotification('Error creating dataset folder: ' + error, 'error');
    }
});

// ==================== Dataset Upload ====================
document.getElementById("uploadDataset")?.addEventListener('click', async () => {
    const datasetName = document.getElementById('datasetNameInput').value;
    if (!datasetName) {
        showNotification('Please enter a dataset name', 'error');
        return;
    }

    const path = repo + workpath + "/" + datasetName + "/";
    const formData = new FormData();
    formData.append('path', path);

    const scanFile = document.getElementById('scanFile').files[0];
    if (scanFile) formData.append('files[]', scanFile, 'scan.txt');

    const bkgFile = document.getElementById('bkgFile').files[0];
    if (bkgFile) formData.append('files[]', bkgFile, 'bkg.png');

    const diffFiles = document.getElementById('diffImages').files;
    Array.from(diffFiles).forEach(file => {
        formData.append('files[]', file);
    });

    try {
        setLoading('uploadDataset');
        await fetch('savefile.php', { method: 'POST', body: formData });
        clearLoading('uploadDataset');
        showNotification('Dataset uploaded successfully', 'success');
    } catch (error) {
        clearLoading('uploadDataset');
        showNotification('Error uploading dataset: ' + error, 'error');
    }
});

// ==================== Process Images ====================
document.getElementById("processImgBtn")?.addEventListener('click', async () => {
    const datasetName = document.getElementById('datasetName').value;
    const outputName = document.getElementById('ptychoOutputName').value;

    if (!datasetName || !outputName) {
        showNotification('Please enter both dataset name and output name', 'error');
        return;
    }

    const nmerge = document.getElementById('nmerge').value;
    const mode = document.getElementById('mode').value;
    const shiftx = document.getElementById('shiftx').value;
    const shifty = document.getElementById('shifty').value;

    const path = repo + workpath + "/";
    const cmd = `./runProcessPtycho.sh ${path} ${datasetName} ${outputName} ${nmerge} ${mode} ${shiftx} ${shifty}`;

    try {
        setLoading('processImgBtn');
        const response = await fetch(url_cmd, { method: 'POST', body: cmd });
        const data = await response.text();
        document.getElementById('processlog').innerText = data;
        clearLoading('processImgBtn');
        displayDiagnosticImages(outputName);
        showNotification('Image processing completed', 'success');
    } catch (error) {
        clearLoading('processImgBtn');
        showNotification('Error during image processing: ' + error, 'error');
    }
});

function displayDiagnosticImages(outputDir) {
    const container = document.getElementById('diagnostic-images-container');
    if (!container) return;

    container.innerHTML = '<div class="loading">Loading diagnostic images...</div>';
    const cmd = `cd ${repo}${workpath} && ls ${outputDir}/*_logimage.jpg ${outputDir}/*_autocorrelation.jpg 2>/dev/null | sort -V`;

    fetch(url_cmd, { method: 'POST', body: cmd })
        .then(response => response.text())
        .then(data => {
            const files = data.trim().split('\n').filter(f => f && f.trim());
            if (files.length === 0) {
                container.innerHTML = '<div class="empty-state">No diagnostic images found</div>';
                return;
            }
            container.innerHTML = files.map(file => {
                const filename = file.split('/').pop();
                const fullPath = `${repo}${workpath}/${outputDir}/${filename}`;
                return `<div class="diagnostic-image-card">
                    <img src="${fullPath}?t=${Date.now()}" alt="${filename}" class="diagnostic-image" onclick="enableInspectMode(this)">
                    <div class="image-label">${filename}</div>
                </div>`;
            }).join('');

            if (!document.getElementById('diagnostic-styles')) {
                const style = document.createElement('style');
                style.id = 'diagnostic-styles';
                style.textContent = `
                    #diagnostic-images-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                    .diagnostic-image-card { background: #f8f9fa; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.08); transition: transform 0.2s; }
                    .diagnostic-image-card:hover { transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
                    .diagnostic-image { width: 100%; height: 150px; object-fit: contain; background: #eee; cursor: zoom-in; }
                    .image-label { padding: 8px; font-size: 12px; text-align: center; color: #555; border-top: 1px solid #eee; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
                    .loading, .empty-state { grid-column: 1 / -1; text-align: center; padding: 20px; color: #888; font-style: italic; }
                    .error { grid-column: 1 / -1; text-align: center; padding: 20px; color: #f44336; }
                `;
                document.head.appendChild(style);
            }
        })
        .catch(err => {
            container.innerHTML = `<div class="error">Failed to load diagnostic images: ${err.message}</div>`;
        });
}

// ==================== View Dataset ====================
document.getElementById("viewDatasetBtn")?.addEventListener('click', async () => {
    const datasetName = document.getElementById('datasetName').value;
    if (!datasetName) {
        showNotification('Please select a dataset name', 'error');
        return;
    }
    window.open(repo + workpath + "/" + datasetName, '_blank');
});

// ==================== Ptychography Reconstruction ====================
document.getElementById("runPtychoBtn")?.addEventListener('click', async () => {
    const btn = document.getElementById('runPtychoBtn');
    const outputName = document.getElementById('ptychoOutputName').value?.trim();
    const path = repo + workpath + "/";

    if (!window.reconstructionState.isRunning && !outputName) {
        showNotification('Enter output directory name first', 'error');
        return;
    }

    // === TERMINATE MODE ===
    if (window.reconstructionState.isRunning) {
        try {
            stopPolling();
            document.getElementById('streamStatus').style.background = '#f44336';
            const killCmd = `./killPtycho.sh "${path}" "${window.reconstructionState.outputName}"`;
            const response = await fetch(url_cmd, { method: 'POST', body: killCmd });
            if (response.ok) {
                showNotification('Reconstruction terminated', 'success');
            }
        } catch (err) {
            showNotification('Termination error (process may have already finished)', 'warning');
        } finally {
            btn.innerHTML = '<i class="layui-icon layui-icon-play"></i> Run Reconstruction';
            btn.className = 'layui-btn layui-btn-danger';
            window.reconstructionState.isRunning = false;
            window.reconstructionState.outputName = null;
            showFallbackVideoMJPEG('ptychoStreamImg', 'ptychoStream',
                repo + workpath + '/' + (outputName || 'out') + '/recon_object.mp4');
        }
        return;
    }

    // === START MODE ===
    window.reconstructionState.isRunning = true;
    window.reconstructionState.outputName = outputName;

    btn.innerHTML = '<i class="layui-icon layui-icon-close"></i> Terminate Reconstruction';
    btn.className = 'layui-btn layui-btn-warm';

    fetch('api_run.php', { method: 'POST', body: `./runPtycho.sh ${path}` })
        .then(resp => resp.json())
        .then(data => {
            if (data.output) {
                document.getElementById('ptycholog').innerText = data.output;
            }
            startPolling(repo + workpath, 'ptycho', 'ptycholog', () => onReconstructionComplete('runPtychoBtn'));
        })
        .catch(err => {
            console.error('Reconstruction start error:', err);
            stopPolling();
            window.reconstructionState.isRunning = false;
            btn.innerHTML = '<i class="layui-icon layui-icon-play"></i> Run Reconstruction';
            btn.className = 'layui-btn layui-btn-danger';
            showFallbackVideoMJPEG('ptychoStreamImg', 'ptychoStream',
                repo + workpath + '/' + (outputName || 'out') + '/recon_object.mp4');
            showNotification('Failed to start reconstruction', 'error');
        });

    showNotification('Reconstruction started! Live stream connecting...', 'success');

    const pageDir = document.URL.replace(/^https?:\/\/[^/]+/, '').replace(/[^/]*$/, '');
    const sockPath = pageDir + repo + workpath + '/cdixx_stream.sock';
    connectMJPEGStream('ptychoStreamImg', 'ptychoStream', sockPath);
});

function onReconstructionComplete(btnId) {
    const btn = document.getElementById(btnId);
    if (btn) {
        btn.innerHTML = '<i class="layui-icon layui-icon-play"></i> Run Reconstruction';
        btn.className = 'layui-btn layui-btn-danger';
    }
    window.reconstructionState.isRunning = false;
    window.reconstructionState.outputName = null;
    document.getElementById('streamStatus').style.background = '#f44336';
    const outputName = document.getElementById('ptychoOutputName').value || 'out';
    showFallbackVideoMJPEG('ptychoStreamImg', 'ptychoStream',
        repo + workpath + '/' + outputName + '/recon_object.mp4');
    loadImage();
    showNotification('Reconstruction completed', 'success');
}

// ==================== Save Configuration ====================
saveTextToWorkingDir("savenote", "ExperimentLog.txt", "note");
saveTextToWorkingDir("savePtychoConfig", "ptycho.cfg", "ptychoConfig");
saveTextToFile("savePtychoConfigDef", "default/ptycho.cfg", "ptychoConfig");
saveTextToFile("savenote_def", "default/ExperimentLog.txt", "note");

// ==================== Autocomplete Initialization ====================
document.addEventListener('DOMContentLoaded', function() {
    const autocompleteIds = [
        {id: "path", dict: "path"},
        {id: "datasetName", dict: "ptycho_dataset"}
    ];
    autocompleteIds.forEach(item => {
        const element = document.getElementById(item.id);
        if (element) autocomplete(element, item.dict);
    });
});

// ==================== Initialize ====================
document.addEventListener('DOMContentLoaded', () => {
    const savedPath = getCookie('ptychoworkpath');
    if (savedPath) {
        workpath = savedPath;
        document.getElementById('path').value = workpath;
        document.getElementById('showpath').textContent = workpath;
        setPath(workpath, {
            'ptycho.cfg': 'ptychoConfig',
            'ExperimentLog.txt': 'note'
        });
        listFiles("cd " + repo + workpath + " ; ls -d */ 2>/dev/null | sed 's/\\\\///g'")
            .then(data => {
                autoFillDictionary["ptycho_dataset"] = data;
            });
    }

    const savedOutput = getCookie('ptycho_outputname');
    if (savedOutput) {
        const outputInput = document.getElementById('ptychoOutputName');
        if (outputInput) outputInput.value = savedOutput;
    }

    // Set fallback video
    const videoEl = document.getElementById('ptychoStream');
    const outputName = document.getElementById('ptychoOutputName').value || 'out';
    videoEl.src = repo + workpath + '/' + outputName + '/recon_object.mp4';
});
