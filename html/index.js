// ==================== CDI-Specific Functions ====================
function loadImage() {
    const timestamp = getTimestamp();
    const path = repo + workpath + "/";

    document.getElementById('recon').src = path + "recon_intensity_cropped.png" + "?t=" + timestamp;
    document.getElementById('recon').title = "CDI reconstruction";
    document.getElementById('recon_phase').src = path + "recon_phase_cropped.png" + "?t=" + timestamp;
    document.getElementById('recon_phase').title = "CDI reconstruction phase";
    document.getElementById('recon_input').src = path + "init_logpattern.jpg" + "?t=" + timestamp;
    document.getElementById('recon_input').title = "CDI input";
    document.getElementById('logimg').src = path + "mergedlog0.jpg" + "?t=" + timestamp;
    document.getElementById('logimg').title = "Monochromatization input";
    document.getElementById('solved').src = path + "solvedlog0.jpg" + "?t=" + timestamp;
    document.getElementById('solved').title = "Solved monochromatic pattern";
    document.getElementById('residual').src = path + "residual_iter_pulseGen.pdf" + "?t=" + timestamp;
    document.getElementById('residual').title = "Residual of monochromatization";
    document.getElementById('residual_img').src = path + "residual_pulseGen.jpg" + "?t=" + timestamp;
    document.getElementById('residual_img').title = "Residual of monochromatization";
    document.getElementById('spectrum_plt').src = path + "spectra.pdf" + "?t=" + timestamp + "#toolbar=0";
    document.getElementById('spectrum_plt').title = "Spectrum";
    document.getElementById('prtf_plt').src = path + "prtf.pdf" + "?t=" + timestamp + "#toolbar=0";
}

// ==================== Image Selection ====================
const cdiImages = ['recon', 'recon_phase', 'recon_input', 'solved', 'residual_img', 'logimg'];
cdiImages.forEach(id => selectListenerWithInspect(id));

// ==================== Event Listeners ====================
document.getElementById("reload")?.addEventListener('click', async () => {
    loadImage();
    showNotification('Images reloaded successfully', 'success');
});

document.getElementById("setpath")?.addEventListener('click', async () => {
    const pathValue = document.getElementById('path').value;
    workpath = pathValue;
    setCookie('cdiworkpath', workpath);

    setPath(pathValue, {
        'pulse.cfg': 'monoconfig',
        'ExperimentLog.txt': 'note',
        'cdi.cfg': 'cdiconfig'
    });

    showNotification('Working folder confirmed', 'success');

    await listFiles("cd " + repo + pathValue + " ; ls -t *.tif 2>/dev/null")
        .then(data => {
            autoFillDictionary["bkg"] = data;
            autoFillDictionary["pattern"] = data;
        });

    await listFiles("cd " + repo + "spectrum" + " ; ls -t *.txt 2>/dev/null")
        .then(data => {
            autoFillDictionary["spect"] = data;
        });
});

// ==================== Upload Functions ====================
document.getElementById("uploadimg")?.addEventListener('click', async () => {
    const path = repo + workpath + "/";
    const files = document.getElementById('background').files;
    const sigfiles = document.getElementById('image').files;
    const formData = new FormData();

    const list = document.getElementsByName("idata");
    let idata = "1";
    for (let i = 0; i < list.length; i++) {
        if (list[i].checked) idata = list[i].value;
    }

    formData.append('path', path);
    document.getElementById('runbkgfile' + idata).value = files[0]?.name || '';
    document.getElementById('runimgfile' + idata).value = sigfiles[0]?.name || '';

    if (files[0]) formData.append('files[]', files[0], files[0].name);
    if (sigfiles[0]) formData.append('files[]', sigfiles[0], sigfiles[0].name);

    try {
        await fetch('savefile.php', { method: 'POST', body: formData });
        await listFiles("cd " + repo + workpath + " ; ls *.tif 2>/dev/null")
            .then(data => {
                autoFillDictionary["bkg"] = data;
                autoFillDictionary["pattern"] = data;
            });
        showNotification('Images uploaded successfully', 'success');
    } catch (error) {
        showNotification('Error uploading images: ' + error, 'error');
    }
});

document.getElementById("uploadspectrum")?.addEventListener('click', async () => {
    const path = repo + "spectrum/";
    const filearray = document.getElementById('spectrumtoupload').files;
    const formData = new FormData();

    if (filearray[0]) {
        document.getElementById('runspectfile').value = filearray[0].name;
    }
    formData.append('path', path);
    Array.from(filearray).forEach(file => {
        formData.append('files[]', file);
    });

    try {
        await fetch('savefile.php', { method: 'POST', body: formData });
        showNotification('Spectrum files uploaded successfully', 'success');
    } catch (error) {
        showNotification('Error uploading spectrum: ' + error, 'error');
    }
});

document.getElementById("uploadfiles")?.addEventListener('click', async () => {
    const path = repo + workpath + "/";
    const filearray = document.getElementById('filestoupload').files;
    const formData = new FormData();
    formData.append('path', path);
    Array.from(filearray).forEach(file => {
        formData.append('files[]', file);
    });

    try {
        await fetch('savefile.php', { method: 'POST', body: formData });
        showNotification('Files uploaded successfully', 'success');
    } catch (error) {
        showNotification('Error uploading files: ' + error, 'error');
    }
});

// ==================== Processing Functions ====================
document.getElementById("prepare")?.addEventListener('click', async () => {
    const list = document.getElementsByName("idata");
    let idata = "1";
    for (let i = 0; i < list.length; i++) {
        if (list[i].checked) idata = list[i].value;
    }

    const path = repo + workpath + "/";
    const cmd = `./runStacker.sh ${path} ${document.getElementById('runbkgfile' + idata).value} ${document.getElementById('runimgfile' + idata).value} ${document.getElementById('nmerge').value} ${document.getElementById('runimgout' + idata).value} ${(document.getElementById('symmetric').checked ? "1" : "0")} ${document.getElementById('shiftx' + idata).value} ${document.getElementById('shifty' + idata).value}`;

    try {
        setLoading('prepare');
        const response = await fetch(url_cmd, { method: 'POST', body: cmd });
        const data = await response.text();
        document.getElementById('preparelog').innerText = data;
        document.getElementById('logimg').src = path + "logimagemerged.jpg?t=" + getTimestamp();
        clearLoading('prepare');
        showNotification('Data preparation completed', 'success');
    } catch (error) {
        clearLoading('prepare');
        showNotification('Error during data preparation: ' + error, 'error');
    }
});

document.getElementById("setspect")?.addEventListener('click', async () => {
    const path = repo + workpath + "/";
    const cmd = `./setSpectrum.sh ${path} ${repo} spectrum/${document.getElementById('runspectfile').value}`;

    try {
        setLoading('setspect');
        await fetch(url_cmd, { method: 'POST', body: cmd });
        document.getElementById('spectrum_plt').src = path + "rawspectrum.pdf?t=" + getTimestamp() + "#toolbar=0";
        clearLoading('setspect');
        showNotification('Spectrum set successfully', 'success');
    } catch (error) {
        clearLoading('setspect');
        showNotification('Error setting spectrum: ' + error, 'error');
    }
});

document.getElementById("compute_prtf")?.addEventListener('click', async () => {
    const path = repo + workpath + "/";
    const cmd = `./compute_prtf.sh ${path}`;

    try {
        await fetch(url_cmd, { method: 'POST', body: cmd });
        document.getElementById('prtf_plt').src = path + "prtf.pdf?t=" + getTimestamp() + "#toolbar=0";
        showNotification('PRTF computed successfully', 'success');
    } catch (error) {
        showNotification('Error computing PRTF: ' + error, 'error');
    }
});

document.getElementById("monochrome_btn")?.addEventListener('click', async () => {
    const path = repo + workpath + "/";

    try {
        setLoading('monochrome_btn');
        const response = await fetch(url_cmd, { method: 'POST', body: `./runPulseGen.sh ${path}` });
        const data = await response.text();

        document.getElementById('monolog').innerText = data;
        const timestamp = getTimestamp();
        document.getElementById('logimg').src = path + "mergedlog0.jpg?t=" + timestamp;
        document.getElementById('solved').src = path + "solvedlog0.jpg?t=" + timestamp;
        document.getElementById('residual').src = path + "residual_iter_pulseGen.pdf?t=" + timestamp;
        document.getElementById('residual_img').src = path + "residual_pulseGen.jpg?t=" + timestamp;
        document.getElementById('spectrum_plt').src = path + "spectra.pdf?t=" + timestamp + "#toolbar=0";

        clearLoading('monochrome_btn');
        showNotification('Monochromatization completed', 'success');
    } catch (error) {
        clearLoading('monochrome_btn');
        showNotification('Error during monochromatization: ' + error, 'error');
    }
});

// ==================== CDI Reconstruction ====================
document.getElementById("cdi_btn")?.addEventListener('click', async () => {
    const btn = document.getElementById('cdi_btn');
    const path = repo + workpath + "/";

    // === TERMINATE MODE ===
    if (window.cdiIsRunning) {
        try {
            stopPolling();
            document.getElementById('streamStatus').style.background = '#f44336';
            await fetch(url_cmd, { method: 'POST', body: `pkill -f "cdi_run" 2>/dev/null; true` });
            showNotification('CDI reconstruction terminated', 'success');
        } catch (err) {
            showNotification('Termination error (process may have already finished)', 'warning');
        } finally {
            btn.innerHTML = '<i class="layui-icon layui-icon-play"></i> Run CDI Reconstruction';
            btn.className = 'layui-btn layui-btn-danger';
            window.cdiIsRunning = false;
            showFallbackVideoMJPEG('cdixxStream', 'cdixxVideo',
                repo + workpath + '/recon_object.mp4');
        }
        return;
    }

    // === START MODE ===
    window.cdiIsRunning = true;
    btn.innerHTML = '<i class="layui-icon layui-icon-close"></i> Terminate CDI';
    btn.className = 'layui-btn layui-btn-warm';

    fetch('api_run.php', { method: 'POST', body: `./runCDI.sh ${path}` })
        .then(resp => resp.json())
        .then(data => {
            if (data.output) {
                document.getElementById('cdilog').innerText = data.output;
            }
            startPolling(repo + workpath, 'cdi', 'cdilog', () => onCDIComplete());
        })
        .catch(err => {
            console.error('CDI start error:', err);
            stopPolling();
            window.cdiIsRunning = false;
            btn.innerHTML = '<i class="layui-icon layui-icon-play"></i> Run CDI Reconstruction';
            btn.className = 'layui-btn layui-btn-danger';
            showFallbackVideoMJPEG('cdixxStream', 'cdixxVideo',
                repo + workpath + '/recon_object.mp4');
            showNotification('Failed to start CDI reconstruction', 'error');
        });

    showNotification('CDI reconstruction started! Live stream connecting...', 'success');

    const pageDir = document.URL.replace(/^https?:\/\/[^/]+/, '').replace(/[^/]*$/, '');
    const sockPath = pageDir + repo + workpath + '/cdixx_stream.sock';
    connectMJPEGStream('cdixxStream', 'cdixxVideo', sockPath);
});

function onCDIComplete() {
    const btn = document.getElementById('cdi_btn');
    if (btn) {
        btn.innerHTML = '<i class="layui-icon layui-icon-play"></i> Run CDI Reconstruction';
        btn.className = 'layui-btn layui-btn-danger';
    }
    window.cdiIsRunning = false;
    document.getElementById('streamStatus').style.background = '#f44336';
    showFallbackVideoMJPEG('cdixxStream', 'cdixxVideo',
        repo + workpath + '/recon_object.mp4');
    loadImage();
    showNotification('CDI reconstruction completed', 'success');
}

// ==================== GPU Selection ====================
document.getElementById("selGPU_btn")?.addEventListener('click', async () => {
    const cmd = `echo export CUDA_VISIBLE_DEVICES=${document.getElementById('iGPU').value} > selectGPU.sh`;
    try {
        await fetch(url_cmd, { method: 'POST', body: cmd });
        showNotification('GPU selection saved', 'success');
    } catch (error) {
        showNotification('Error setting GPU: ' + error, 'error');
    }
});

// ==================== Save Configuration Files ====================
saveTextToWorkingDir("savepulse", "pulse.cfg", "monoconfig");
saveTextToWorkingDir("savenote", "ExperimentLog.txt", "note");
saveTextToWorkingDir("savecdi", "cdi.cfg", "cdiconfig");
saveTextToFile("savepulse_def", "default/pulse.cfg", "monoconfig");
saveTextToFile("savenote_def", "default/ExperimentLog.txt", "note");
saveTextToFile("savecdi_def", "default/cdi.cfg", "cdiconfig");

// ==================== Autocomplete Initialization ====================
document.addEventListener('DOMContentLoaded', function() {
    const autocompleteIds = [
        {id: "runbkgfile1", dict: "bkg"},
        {id: "runimgfile1", dict: "pattern"},
        {id: "runbkgfile2", dict: "bkg"},
        {id: "runimgfile2", dict: "pattern"},
        {id: "path", dict: "path"},
        {id: "runspectfile", dict: "spect"}
    ];
    autocompleteIds.forEach(item => {
        const element = document.getElementById(item.id);
        if (element) autocomplete(element, item.dict);
    });
});

// ==================== Initialize ====================
document.addEventListener('DOMContentLoaded', () => {
    if (workpath) {
        document.getElementById('path').value = workpath;
        document.getElementById('showpath').textContent = workpath;
        setPath(workpath, {
            'pulse.cfg': 'monoconfig',
            'ExperimentLog.txt': 'note',
            'cdi.cfg': 'cdiconfig'
        });
        const videoEl = document.getElementById('cdixxVideo');
        videoEl.src = repo + workpath + '/recon_object.mp4';
    }
});
