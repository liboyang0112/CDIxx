// ==================== Configuration ====================
const nodeurl = window.location.origin + ':8080';
const url_cmd = "runcommand.php";
const repo = 'images/';
const isPtychoPage = window.location.pathname.includes('ptycho');
const cookieName = isPtychoPage ? 'ptychoworkpath' : 'cdiworkpath';

// Get workpath from the appropriate cookie
let workpath = document.cookie.split("; ").find((row) => row.startsWith(cookieName + "="))?.split("=")[1];

window.hlsInstances = window.hlsInstances || {};
window.reconstructionState = {
  isRunning: false,
  streamTimeout: null,
  outputName: null
};


let autoFillDictionary = {
    "path": {},
    "bkg": {},
    "pattern": {},
    "spect": {},
    "ptycho_dataset": {}
};

const imageNames = {
    'recon': 'CDI Reconstruction',
    'recon_phase': 'Reconstruction Phase',
    'recon_input': 'CDI Input',
    'solved': 'Solved Pattern',
    'residual_img': 'Residual Image',
    'logimg': 'Log Image',
    'recon_pupil_proj': 'Pupil Projection',
    'recon_pupil': 'Recon Pupil',
    'recon_pupil_b4_proj': 'Pupil Before Projection',
    'ptycho_afterIterwave': 'Wave After Iteration',
    'ptycho_probe_afterIter': 'Probe After Iteration'
};

// ==================== Utility Functions ====================
function showNotification(content, type = 'info') {
    const icon = type === 'success' ? 1 : (type === 'error' ? 2 : 0);
    layui.layer.msg(content, {
        icon: icon,
        time: 2000,
        shade: 0,  // Remove blocking overlay
        offset: '15px',  // Position at top
        anim: 1,  // Slide down animation
        shadeClose: false
    });
}

function setLoading(buttonId) {
    const button = document.getElementById(buttonId);
    if (button) button.disabled = true;
    layui.layer.load(1, {shade: [0.3, '#000']});
}

function clearLoading(buttonId) {
    const button = document.getElementById(buttonId);
    if (button) button.disabled = false;
    layui.layer.closeAll('loading');
}

function getTimestamp() {
    return new Date().getTime();
}

// ==================== File Operations ====================
function listFiles(comd) {
    return fetch(url_cmd, {
        method: 'POST',
        body: comd,
    }).then(response => {
        return response.text();
    }).then(data => {
        // Filter out empty lines and common unwanted entries
        return data.split("\n")
            .filter(item => item.trim() !== '')
            .filter(item => !item.includes('Permission denied'))
            .filter(item => !item.includes('No such file'))
            .filter(item => !item.includes('cannot access'));
    });
}

function saveTextToFile(buttonID, filename, contentId) {
    const button = document.getElementById(buttonID);
    button.addEventListener('click', async () => {
        button.disabled = true;
        const url_write = "writefile.php/";
        try {
            const response = await fetch(url_write + filename, {
                method: 'POST',
                body: document.getElementById(contentId).value
            });
            const data = await response.text();
            button.disabled = false;
            console.log(data);
            showNotification('File saved successfully', 'success');
        } catch (error) {
            button.disabled = false;
            showNotification('Error saving file: ' + error, 'error');
        }
    });
}

function saveTextToWorkingDir(buttonID, filename, contentId, basePath = repo) {
    const button = document.getElementById(buttonID);
    button.addEventListener('click', async () => {
        button.disabled = true;
        const url_write = "writefile.php/";
        const path = basePath + (document.getElementById('path')?.value || workpath);
        try {
            const response = await fetch(url_write + path + "/" + filename, {
                method: 'POST',
                body: document.getElementById(contentId).value
            });
            const data = await response.text();
            button.disabled = false;
            console.log(data);
            showNotification('File saved to working directory', 'success');
        } catch (error) {
            button.disabled = false;
            showNotification('Error saving file: ' + error, 'error');
        }
    });
}

// ==================== Path Management ====================
function setPath(path, configFiles = {}) {
    const url_path = "setpath.php";
    const url_read = "readfile.php";
    
    if (!path || path.trim() === "") {
        document.getElementById('showpath').innerText = "Path not filled, please fill the text before click confirm";
        showNotification('Please enter a valid folder path', 'error');
        return;
    }
    
    path = repo + path;
    const ele = document.getElementById('showpath');
    ele.href = path;
    ele.innerText = "Working Path: " + path;
    
    // Save to the correct cookie
    document.cookie = cookieName + '=' + path.replace(repo, '') + '; path=/';
    
    fetch(url_path, {
        method: 'POST',
        body: path
    }).then(() => {
        // Load configuration files
        Object.entries(configFiles).forEach(([filename, elementId]) => {
            fetch(url_read, {
                method: 'POST',
                body: path + "/" + filename
            }).then(response => response.text())
              .then(data => {
                  if (document.getElementById(elementId)) {
                      document.getElementById(elementId).value = data;
                  }
              })
              .catch(error => console.error('Error loading config:', filename, error));
        });
    });
}

// ==================== Autocomplete ====================
function autocomplete(inp, dictName) {
    let currentFocus;
    
    inp.addEventListener("input", function() {
        const arr = autoFillDictionary[dictName];
        let a, b, i, val = this.value;
        closeAllLists();
        
        if (!val) return false;
        
        currentFocus = -1;
        a = document.createElement("DIV");
        a.setAttribute("id", this.id + "autocomplete-list");
        a.setAttribute("class", "autocomplete-items");
        this.parentNode.appendChild(a);
        
        let matches = 0;
        for (i = 0; i < arr.length; i++) {
            if (arr[i] && arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
                matches++;
                b = document.createElement("DIV");
                b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
                b.innerHTML += arr[i].substr(val.length);
                b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
                b.addEventListener("click", function(e) {
                    inp.value = this.getElementsByTagName("input")[0].value;
                    closeAllLists();
                });
                a.appendChild(b);
            }
        }
        
        if (matches === 0) {
            b = document.createElement("DIV");
            b.innerHTML = "<em>No matches found</em>";
            b.style.padding = "8px 12px";
            b.style.color = "#999";
            a.appendChild(b);
        }
    });
    
    inp.addEventListener("keydown", function(e) {
        const x = document.getElementById(this.id + "autocomplete-list");
        if (!x) return;
        
        const items = x.getElementsByTagName("div");
        if (e.keyCode == 40) {
            currentFocus++;
            addActive(items);
        } else if (e.keyCode == 38) {
            currentFocus--;
            addActive(items);
        } else if (e.keyCode == 13) {
            e.preventDefault();
            if (currentFocus > -1 && items[currentFocus]) {
                items[currentFocus].click();
            }
        }
    });
    
    function addActive(x) {
        if (!x) return false;
        removeActive(x);
        if (currentFocus >= x.length) currentFocus = 0;
        if (currentFocus < 0) currentFocus = (x.length - 1);
        x[currentFocus].classList.add("autocomplete-active");
    }
    
    function removeActive(x) {
        for (let i = 0; i < x.length; i++) {
            x[i].classList.remove("autocomplete-active");
        }
    }
    
    function closeAllLists(elmnt) {
        const x = document.getElementsByClassName("autocomplete-items");
        for (let i = 0; i < x.length; i++) {
            if (elmnt != x[i] && elmnt != inp) {
                x[i].parentNode.removeChild(x[i]);
            }
        }
    }
    
    document.addEventListener("click", function(e) {
        closeAllLists(e.target);
    });
}

// ==================== Image Selection & Transformation ====================
let degree = 0;
let flip = 1;
let operateImage = "recon";

function selectListenerWithInspect(imageid) {
    const imgElement = document.getElementById(imageid);
    if (!imgElement) return;
    
    // Track double-click state
    let clickTimer = null;
    const DOUBLE_CLICK_DELAY = 20;
    
    imgElement.addEventListener('click', function(e) {
        // Prevent if clicking on controls
        if (e.target.closest('.floating-controls')) return;
        
        // Clear any existing timer
        if (clickTimer) {
            clearTimeout(clickTimer);
            clickTimer = null;
            return; // This is a double-click, handled below
        }
        
        // Set timer for single click
        clickTimer = setTimeout(() => {
            clickTimer = null;
            
            // Single click: select image
            document.querySelectorAll('.image-container img').forEach(img => {
                img.classList.remove('selected');
            });
            this.classList.add('selected');
            
            operateImage = imageid;
            if (document.getElementById('active-image-name')) {
                document.getElementById('active-image-name').textContent = imageNames[imageid] || imageid;
            }
        }, DOUBLE_CLICK_DELAY);
    });
    
    imgElement.addEventListener('dblclick', function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // Clear the single-click timer
        if (clickTimer) {
            clearTimeout(clickTimer);
            clickTimer = null;
        }
        
        // Double click: open preview
        enableInspectMode(this);
    });
}

function resetImageTransforms() {
    document.querySelectorAll('.image-container img').forEach(img => {
        img.style.transform = "";
    });
}

// ==================== Image Inspection Mode ====================
let inspectMode = false;
let inspectImageElement = null;
let offsetX = 0, offsetY = 0;
let isDragging = false;
let dragStartX = 0, dragStartY = 0;

function enableInspectMode(imgElement) {
    const overlay = document.getElementById('inspectOverlay');
    inspectImageElement = document.getElementById('inspectImage');
    
    if (!overlay || !inspectImageElement) return;
    
    overlay.style.display = 'block';
    inspectMode = true;
    
    inspectImageElement.onload = function() {
        positionImageCenter();
    };
    
    inspectImageElement.src = imgElement.src;
    
    overlay.addEventListener('mousedown', startDrag);
    document.addEventListener('mousemove', dragImage);
    document.addEventListener('mouseup', stopDrag);
    document.addEventListener('keydown', handleKeydown);
    
    const closeBtn = document.getElementById('closeInspect');
    if (closeBtn) {
        closeBtn.onclick = disableInspectMode;
    }
}

function disableInspectMode() {
    const overlay = document.getElementById('inspectOverlay');
    if (!overlay) return;
    
    overlay.style.display = 'none';
    inspectMode = false;
    
    overlay.removeEventListener('mousedown', startDrag);
    document.removeEventListener('mousemove', dragImage);
    document.removeEventListener('mouseup', stopDrag);
    document.removeEventListener('keydown', handleKeydown);
    
    const closeBtn = document.getElementById('closeInspect');
    if (closeBtn) {
        closeBtn.onclick = null;
    }
}

function positionImageCenter() {
    if (!inspectImageElement || !inspectImageElement.naturalWidth) return;
    
    const containerWidth = window.innerWidth;
    const containerHeight = window.innerHeight;
    const imgWidth = inspectImageElement.naturalWidth;
    const imgHeight = inspectImageElement.naturalHeight;
    
    const scaleX = containerWidth / imgWidth;
    const scaleY = containerHeight / imgHeight;
    const scale = Math.max(scaleX, scaleY, 1);
    
    offsetX = (containerWidth - imgWidth * scale) / 2;
    offsetY = (containerHeight - imgHeight * scale) / 2;
    updateImageTransform();
}

function startDrag(e) {
    if (e.target.id === 'closeInspect') return;
    e.preventDefault();
    isDragging = true;
    dragStartX = e.clientX - offsetX;
    dragStartY = e.clientY - offsetY;
    document.getElementById('inspectOverlay').style.cursor = 'grabbing';
}

function dragImage(e) {
    if (!isDragging || !inspectMode) return;
    e.preventDefault();
    offsetX = e.clientX - dragStartX;
    offsetY = e.clientY - dragStartY;
    updateImageTransform();
}

function stopDrag() {
    isDragging = false;
    if (document.getElementById('inspectOverlay')) {
        document.getElementById('inspectOverlay').style.cursor = 'grab';
    }
}

function handleKeydown(e) {
    if (e.key === 'Escape') {
        disableInspectMode();
    }
}

function updateImageTransform() {
    if (!inspectImageElement || !inspectImageElement.naturalWidth) return;
    
    const containerWidth = window.innerWidth;
    const containerHeight = window.innerHeight;
    const imgWidth = inspectImageElement.naturalWidth;
    const imgHeight = inspectImageElement.naturalHeight;
    
    const scaleX = containerWidth / imgWidth;
    const scaleY = containerHeight / imgHeight;
    const scale = Math.max(scaleX, scaleY, 1);
    
    inspectImageElement.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
}

// ==================== Image Transformations ====================
function rotateImage(degrees) {
    degree = (degree + degrees) % 360;
    applyTransform();
    showNotification(`Rotated to ${degree}°`, 'success');
}

function transposeImage() {
    degree = (degree + 180) % 360;
    applyTransform();
    showNotification('Transposed', 'success');
}

function flipImage() {
    flip *= -1;
    applyTransform();
    showNotification('Flipped horizontally', 'success');
}

function resetImage() {
    flip = 1;
    degree = 0;
    applyTransform();
    if (document.getElementById(operateImage)) {
        document.getElementById(operateImage).classList.add('selected');
    }
    showNotification('Image reset to original state', 'success');
}

function applyTransform() {
    const img = document.getElementById(operateImage);
    if (img) {
        img.style.transform = `rotate(${degree}deg) scaleX(${flip})`;
    }
}

// ==================== HLS Video Streaming ====================
// FIXED HLS FUNCTION - Properly destroys/recreates stream
function setupHLSStream(videoElementId, streamUrl, latencyElementId = null, statusElementId = null) {
  const video = document.getElementById(videoElementId);
  if (!video) return;

  // DESTROY EXISTING INSTANCE (critical fix)
  if (window.hlsInstances[videoElementId]) {
    window.hlsInstances[videoElementId].destroy();
    delete window.hlsInstances[videoElementId];
    video.src = '';
    video.load();
  }

  // Reset status indicator
  const statusEl = statusElementId ? document.getElementById(statusElementId) : null;
  if (statusEl) {
    statusEl.style.background = '#f44336';
    statusEl.style.animation = 'pulse 1.5s infinite';
  }

  // Initialize NEW stream
  if (Hls.isSupported()) {
    const hls = new Hls({
      enableWorker: true,
      lowLatencyMode: true,
      backBufferLength: 3,
      maxBufferLength: 6
    });
    
    hls.loadSource(streamUrl + `?_=${Date.now()}`); // Cache-busting
    hls.attachMedia(video);
    
    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      if (statusEl) {
        statusEl.style.background = '#4caf50';
        statusEl.style.animation = 'none';
      }
      video.play().catch(e => console.log('Autoplay warning:', e));
    });
    
    hls.on(Hls.Events.ERROR, (event, data) => {
      if (data.fatal) hls.destroy();
    });
    
    window.hlsInstances[videoElementId] = hls;
    
    // Latency monitor
    if (latencyElementId) {
      setInterval(() => {
        if (hls.liveSyncPosition && video.currentTime) {
          const latency = hls.liveSyncPosition - video.currentTime;
          document.getElementById(latencyElementId).textContent = latency.toFixed(1);
        }
      }, 1000);
    }
  } 
  // Safari fallback handled in original function - omitted for brevity
}

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Layui
    layui.use(['form', 'element', 'layer'], function() {
        const form = layui.form;
        const element = layui.element;
        
        form.render();
        element.init();
    });
    
    // Load path autocomplete data from images directory
    listFiles("cd " + repo + " ; find * -maxdepth 0 -type d 2>/dev/null")
        .then(data => {
            autoFillDictionary["path"] = data;
            console.log('Paths loaded:', data);
        })
        .catch(error => {
            console.error('Error loading paths:', error);
        });
    
    // Toggle floating controls
    const toggleBtn = document.getElementById('toggle-floating-btn');
    const floatingControls = document.querySelector('.floating-controls');
    const closeFloatingBtn = document.getElementById('close-floating-btn');
    
    if (toggleBtn && floatingControls) {
        toggleBtn.addEventListener('click', function() {
            if (floatingControls.style.display === 'none' || floatingControls.style.display === '') {
                floatingControls.style.display = 'flex';
                toggleBtn.innerHTML = '<i class="layui-icon layui-icon-up"></i> Hide Image Tools';
            } else {
                floatingControls.style.display = 'none';
                toggleBtn.innerHTML = '<i class="layui-icon layui-icon-down"></i> Show Image Tools';
            }
        });
    }
    
    if (closeFloatingBtn && floatingControls) {
        closeFloatingBtn.addEventListener('click', function() {
            floatingControls.style.display = 'none';
            if (toggleBtn) {
                toggleBtn.innerHTML = '<i class="layui-icon layui-icon-down"></i> Show Image Tools';
            }
        });
    }
    
    // Image transformation buttons
    const rotateBtn = document.getElementById('rotate_btn');
    const rotate180Btn = document.getElementById('rotate180_btn');
    const flipBtn = document.getElementById('flip_btn');
    const resetBtn = document.getElementById('reset_btn');
    
    if (rotateBtn) rotateBtn.addEventListener('click', () => rotateImage(90));
    if (rotate180Btn) rotate180Btn.addEventListener('click', () => transposeImage());
    if (flipBtn) flipBtn.addEventListener('click', () => flipImage());
    if (resetBtn) resetBtn.addEventListener('click', () => resetImage());
});
