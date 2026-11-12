// ==================== Page-Specific Initialization ====================
const ssptychoImages = [
    'recon_object_wave',
    'recon_phase',
    'recon_intensity',
    'obj_freq',
    'recon_probe_wave'
];

ssptychoImages.forEach(id => {
    selectListenerWithInspect(id)
});

if (document.getElementById('recon_object_wave')) {
    document.getElementById('recon_object_wave').classList.add('selected');
    operateImage = 'recon_object_wave';
    if (document.getElementById('active-image-name')) {
        document.getElementById('active-image-name').textContent = imageNames['recon_object_wave'];
    }
}

// ==================== Load Images ====================
function loadImage(){
    const outputName = document.getElementById('runimgout').value?.trim();
    if (!outputName) {
        showNotification('Please enter output directory name', 'error');
        return;
    }

    setCookie('ssptycho_outputname', outputName);
    console.log('[loadImage] Saved output name to cookie:', outputName);

    const timestamp = getTimestamp();
    const path = repo + workpath + "/" + outputName + "/";

    ssptychoImages.forEach(imgId => {
        const img = document.getElementById(imgId);
        if (img) {
            img.src = path + imgId + ".jpg?t=" + timestamp;
        }
    });
    displayDiagnosticImages(outputName);
    showNotification('Images loaded from output directory', 'success');
}

// ==================== Reconstruction Complete Handler ====================
function onReconstructionComplete() {
    const btn = document.getElementById('runSSPtychoBtn');
    if (btn) {
        btn.innerHTML = '<i class="layui-icon layui-icon-play"></i> Run Reconstruction';
        btn.className = 'layui-btn layui-btn-danger';
    }
    window.reconstructionState.isRunning = false;
    window.reconstructionState.outputName = null;
    document.getElementById('streamStatus').style.background = '#f44336';
    const outputName = document.getElementById('runimgout').value || 'out';
    showFallbackVideoMJPEG('ssptychoStream', 'ssptychoStreamVideo',
        repo + workpath + '/' + outputName + '/recon_object.mp4');
    loadImage();
    showNotification('Reconstruction completed', 'success');
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
    setCookie('ssptychoworkpath', workpath);

    setPath(pathValue, {
        'ssptycho.cfg': 'ssptychoConfig',
        'ExperimentLog.txt': 'note'
    });

    showNotification('Working folder confirmed: ' + workpath, 'success');

    // Load autocomplete data for dataset folders
    await listFiles("cd " + repo + workpath + " ; ls -d */ 2>/dev/null | sed 's/\\///g'")
        .then(data => {
            autoFillDictionary["dataset"] = data;
            console.log('Datasets loaded:', data);
        });
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
        const response = await fetch(url_cmd, { method: 'POST', body: cmd });
        const data = await response.text();

        showNotification(`Dataset folder created: ${datasetName}`, 'success');

        await listFiles("cd " + repo + workpath + " ; ls -d */ 2>/dev/null | sed 's/\\///g'")
            .then(data => {
                autoFillDictionary["dataset"] = data;
            });
    } catch (error) {
        showNotification('Error creating dataset folder: ' + error, 'error');
    }
});

// ==================== Upload ====================
document.getElementById("uploadimg")?.addEventListener('click', async () => {
    const datasetName = document.getElementById('datasetNameInput').value;
    if (!datasetName) {
        showNotification('Please enter a dataset name first', 'error');
        return;
    }

    const path = repo + workpath + "/" + datasetName + "/";
    const url = 'savefile.php';
    const filearray = document.getElementById('imagestoupload').files;
    const formData = new FormData();

    if (filearray.length === 0) {
        showNotification('Please select files to upload', 'error');
        return;
    }

    // Ensure dataset directory exists
    await fetch(url_cmd, { method: 'POST', body: `mkdir -p ${repo}${workpath}/${datasetName}` });

    formData.append('path', path);
    Array.from(filearray).forEach(file => {
        formData.append('files[]', file);
    });

    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        const data = await response.text();
        console.log(data);

        await listFiles("cd " + repo + workpath + "/" + datasetName + " ; ls *.tif *.png 2>/dev/null")
            .then(data => {
                autoFillDictionary["bkg"] = data;
                autoFillDictionary["pattern"] = data;
            });

        showNotification(filearray.length + ' file(s) uploaded successfully', 'success');
    } catch (error) {
        showNotification('Error uploading files: ' + error, 'error');
    }
});

// ==================== Process Experimental Data ====================
document.getElementById("prepare")?.addEventListener('click', async () => {
    const datasetName = document.getElementById('datasetName').value;
    const outputName = document.getElementById('runimgout').value;
    const bkg = document.getElementById('runbkgfile').value;
    const pattern = document.getElementById('runimgfile').value;
    const gratings = document.getElementById('rungratingfile').value;
    const nmerge = document.getElementById('nmerge').value;
    const symmetric = document.getElementById('symmetric').checked ? "1" : "0";
    const shiftx = document.getElementById('shiftx').value;
    const shifty = document.getElementById('shifty').value;

    if (!datasetName || !outputName) {
        showNotification('Please enter both dataset name and output name', 'error');
        return;
    }
    if (!bkg || !pattern || !gratings) {
        showNotification('Please fill in Background, Pattern, and Gratings file names', 'error');
        return;
    }

    try {
        setLoading('prepare');

        const path = repo + workpath + "/";
        const dsPath = path + datasetName + "/";

        // Process pattern - background
        const cmd1 = `./runStacker.sh ${dsPath} ${bkg} ${pattern} ${nmerge} ${outputName} ${symmetric} ${shiftx} ${shifty}`;
        const resp1 = await fetch(url_cmd, { method: 'POST', body: cmd1 });
        const data1 = await resp1.text();

        // Process gratings - background
        const cmd2 = `./runStacker.sh ${dsPath} ${bkg} ${gratings} ${nmerge} ${outputName} ${symmetric} ${shiftx} ${shifty}`;
        const resp2 = await fetch(url_cmd, { method: 'POST', body: cmd2 });
        const data2 = await resp2.text();

        document.getElementById('preparelog').innerText = `[Pattern]\n${data1}\n\n[Gratings]\n${data2}`;
        clearLoading('prepare');
        displayDiagnosticImages(outputName);
        showNotification('Data preparation completed', 'success');
    } catch (error) {
        clearLoading('prepare');
        showNotification('Error during data preparation: ' + error, 'error');
    }
});

function displayDiagnosticImages(outputName) {
    const container = document.getElementById('diagnostic-images-container');
    if (!container || !outputName) return;

    container.innerHTML = '<div class="loading">Loading diagnostic images...</div>';

    const path = repo + workpath + "/" + outputName;
    const cmd = `cd ${repo}${workpath} && ls ${path}/*_logimage.jpg ${path}/*_autocorrelation.jpg 2>/dev/null | sort -V`;

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
                const fullPath = `${repo}${workpath}/${outputName}/${filename}`;
                return `
                    <div class="diagnostic-image-card">
                        <img src="${fullPath}?t=${Date.now()}"
                             alt="${filename}"
                             class="diagnostic-image"
                             onclick="enableInspectMode(this)">
                        <div class="image-label">${filename}</div>
                    </div>
                `;
            }).join('');

            if (!document.getElementById('diagnostic-styles')) {
                const style = document.createElement('style');
                style.id = 'diagnostic-styles';
                style.textContent = `
                    #diagnostic-images-container {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 15px;
                        margin: 20px 0;
                    }
                    .diagnostic-image-card {
                        background: #f8f9fa;
                        border-radius: 6px;
                        overflow: hidden;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                        transition: transform 0.2s;
                    }
                    .diagnostic-image-card:hover {
                        transform: translateY(-3px);
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    }
                    .diagnostic-image {
                        width: 100%;
                        height: 150px;
                        object-fit: contain;
                        background: #eee;
                        cursor: zoom-in;
                    }
                    .image-label {
                        padding: 8px;
                        font-size: 12px;
                        text-align: center;
                        color: #555;
                        border-top: 1px solid #eee;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }
                    .loading, .empty-state {
                        grid-column: 1 / -1;
                        text-align: center;
                        padding: 20px;
                        color: #888;
                        font-style: italic;
                    }
                    .error {
                        grid-column: 1 / -1;
                        text-align: center;
                        padding: 20px;
                        color: #f44336;
                    }
                `;
                document.head.appendChild(style);
            }
        })
        .catch(err => {
            console.error('Error loading diagnostic images:', err);
            container.innerHTML = `<div class="error">Failed to load diagnostic images: ${err.message}</div>`;
        });
}

// ==================== SSPtycho Reconstruction ====================
document.getElementById("runSSPtychoBtn")?.addEventListener('click', async () => {
    const btn = document.getElementById('runSSPtychoBtn');
    const outputName = document.getElementById('runimgout').value?.trim();
    const path = repo + workpath + "/";

    if (!window.reconstructionState.isRunning && !outputName) {
        showNotification('Enter output name first', 'error');
        return;
    }

    // === TERMINATE MODE ===
    if (window.reconstructionState.isRunning) {
        try {
            stopPolling();
            document.getElementById('streamStatus').style.background = '#f44336';
            const killCmd = `pkill -f "ssptycho_run" 2>/dev/null; true`;
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
            showFallbackVideoMJPEG('ssptychoStream', 'ssptychoStreamVideo',
                repo + workpath + '/' + (outputName || 'out') + '/recon_object.mp4');
        }
        return;
    }

    // === START MODE ===
    window.reconstructionState.isRunning = true;
    window.reconstructionState.outputName = outputName;

    btn.innerHTML = '<i class="layui-icon layui-icon-close"></i> Terminate Reconstruction';
    btn.className = 'layui-btn layui-btn-warm';

    fetch('api_run.php', { method: 'POST', body: `./runSSPtycho.sh ${path}` })
        .then(resp => resp.json())
        .then(data => {
            if (data.output) {
                document.getElementById('ssptycholog').innerText = data.output;
            }
            startPolling(repo + workpath, 'ssptycho', 'ssptycholog', onReconstructionComplete);
        })
        .catch(err => {
            console.error('Reconstruction start error:', err);
            stopPolling();
            window.reconstructionState.isRunning = false;
            btn.innerHTML = '<i class="layui-icon layui-icon-play"></i> Run Reconstruction';
            btn.className = 'layui-btn layui-btn-danger';
            showFallbackVideoMJPEG('ssptychoStream', 'ssptychoStreamVideo',
                repo + workpath + '/' + (outputName || 'out') + '/recon_object.mp4');
            showNotification('Failed to start reconstruction', 'error');
        });

    showNotification('Reconstruction started! Live stream connecting...', 'success');

    const pageDir = document.URL.replace(/^https?:\/\/[^/]+/, '').replace(/[^/]*$/, '');
    const sockPath = pageDir + repo + workpath + '/cdixx_stream.sock';
    connectMJPEGStream('ssptychoStream', 'ssptychoStreamVideo', sockPath);
});

// ==================== GPU Selection ====================
document.getElementById("selGPU_btn")?.addEventListener('click', async () => {
    const cmd = `echo export CUDA_VISIBLE_DEVICES=${document.getElementById('iGPU').value} > selectGPU.sh`;
    try {
        const response = await fetch(url_cmd, {
            method: 'POST',
            body: cmd
        });
        const data = await response.text();
        console.log(data);
        showNotification('GPU selection saved', 'success');
    } catch (error) {
        showNotification('Error setting GPU: ' + error, 'error');
    }
});

// ==================== Save Configuration ====================
saveTextToWorkingDir("savenote", "ExperimentLog.txt", "note");
saveTextToWorkingDir("saveSSPtychoConfig", "ssptycho.cfg", "ssptychoConfig");
saveTextToFile("saveSSPtychoConfigDef", "default/ssptycho.cfg", "ssptychoConfig");
saveTextToFile("savenote_def", "default/ExperimentLog.txt", "note");

// ==================== Autocomplete Initialization ====================
document.addEventListener('DOMContentLoaded', function() {
    const autocompleteIds = [
        {id: "path", dict: "path"},
        {id: "runbkgfile", dict: "bkg"},
        {id: "runimgfile", dict: "pattern"},
        {id: "rungratingfile", dict: "pattern"}
    ];

    autocompleteIds.forEach(item => {
        const element = document.getElementById(item.id);
        if (element) {
            autocomplete(element, item.dict);
        }
    });
});

// ==================== Initialize ====================
document.addEventListener('DOMContentLoaded', () => {
    const savedPath = getCookie('ssptychoworkpath');
    if (savedPath) {
        workpath = savedPath;
        document.getElementById('path').value = workpath;
        document.getElementById('showpath').textContent = workpath;
        setPath(workpath, {
            'ssptycho.cfg': 'ssptychoConfig',
            'ExperimentLog.txt': 'note'
        });
        listFiles("cd " + repo + workpath + " ; ls -d */ 2>/dev/null | sed 's/\\///g'")
            .then(data => {
                autoFillDictionary["dataset"] = data;
            });
        console.log('workpath restored from cookie:', workpath);
    }

    // Restore output name
    const savedOutput = getCookie('ssptycho_outputname');
    if (savedOutput) {
        const outputInput = document.getElementById('runimgout');
        if (outputInput) {
            outputInput.value = savedOutput;
            console.log('output name restored from cookie:', savedOutput);
        }
    }

    // Set fallback video (recon_object.mp4) when not streaming
    const videoEl = document.getElementById('ssptychoStreamVideo');
    const outputName = document.getElementById('runimgout').value || 'out';
    videoEl.src = repo + workpath + '/' + outputName + '/recon_object.mp4';
});
