/* ═══════════════════════════════════════════════════════════
   RAG Answering Service — Frontend Application
   ═══════════════════════════════════════════════════════════ */

// ── State ──────────────────────────────────────────────────
const state = {
    currentTab: 'query',
    qaPage: 1,
    qaPerPage: 20,
    deleteTargetId: null,
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    selectedFile: null,
    selectedFileType: null, // 'audio', 'video', 'ingest'
};

// ── Toast Notifications ────────────────────────────────────
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
}

// ── API Client ─────────────────────────────────────────────
const API = {
    async ask(question, language, subject, topic) {
        const resp = await fetch('/api/v1/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: question, language: language || 'en', subject, topic }),
        });
        if (!resp.ok) {
            const errText = await resp.text();
            try { throw new Error(JSON.parse(errText).detail || resp.statusText); } 
            catch(e) { throw new Error(errText || resp.statusText); }
        }
        return resp.json();
    },

    async askAudio(file, language, subject, topic) {
        const form = new FormData();
        form.append('file', file);
        if (language) form.append('language', language);
        if (subject) form.append('subject', subject);
        if (topic) form.append('topic', topic);
        const resp = await fetch('/api/v1/ask/audio', { method: 'POST', body: form });
        if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
        return resp.json();
    },

    async askVideo(file, language, subject, topic) {
        const form = new FormData();
        form.append('file', file);
        if (language) form.append('language', language);
        if (subject) form.append('subject', subject);
        if (topic) form.append('topic', topic);
        const resp = await fetch('/api/v1/ask/video', { method: 'POST', body: form });
        if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
        return resp.json();
    },

    async ingestFile(file, subject, topic, language) {
        const form = new FormData();
        form.append('file', file);
        if (subject) form.append('subject', subject);
        if (topic) form.append('topic', topic);
        if (language) form.append('language', language);
        const resp = await fetch('/api/v1/ingest/file', { method: 'POST', body: form });
        if (!resp.ok) {
            const errText = await resp.text();
            try { throw new Error(JSON.parse(errText).detail || resp.statusText); } 
            catch(e) { throw new Error(errText || resp.statusText); }
        }
        return resp.json();
    },

    async checkIngestStatus(documentId) {
        const resp = await fetch(`/api/v1/ingest/status/${documentId}`);
        if (!resp.ok) throw new Error('Status fetch failed');
        return resp.json();
    },

    async listDocuments() {
        const resp = await fetch('/api/v1/ingest/documents');
        if (!resp.ok) throw new Error('Failed to fetch documents');
        return resp.json();
    },

    async deleteDocument(documentId) {
        const resp = await fetch(`/api/v1/ingest/documents/${documentId}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error('Failed to delete document');
        return resp.json();
    },

    async listQA(page = 1, perPage = 20) {
        const resp = await fetch(`/api/v1/qa?page=${page}&per_page=${perPage}`);
        if (!resp.ok) throw new Error('Failed to fetch Q/A pairs');
        return resp.json();
    },

    async createQA(question, answer, language) {
        const resp = await fetch('/api/v1/qa', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer, language: language || 'en' }),
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
        return resp.json();
    },

    async updateQA(id, question, answer, editReason) {
        const resp = await fetch(`/api/v1/qa/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer, edit_reason: editReason }),
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
        return resp.json();
    },

    async deleteQA(id) {
        const resp = await fetch(`/api/v1/qa/${id}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error('Failed to delete');
        return resp.json();
    },

    async compare(question, userAnswer) {
        const resp = await fetch('/api/v1/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question, user_answer: userAnswer })
        });
        if (!resp.ok) {
            const errText = await resp.text();
            try { throw new Error(JSON.parse(errText).detail || resp.statusText); } 
            catch(e) { throw new Error(errText || resp.statusText); }
        }
        return resp.json();
    },

    async getHealth() {
        const resp = await fetch('/health');
        if (!resp.ok) throw new Error('Health check failed');
        return resp.json();
    },

    async getMetrics() {
        const resp = await fetch('/metrics');
        if (!resp.ok) throw new Error('Metrics fetch failed');
        return resp.json();
    },
};

// ── Tab Navigation ─────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        state.currentTab = tab;

        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.getElementById(`tab-${tab}`).classList.add('active');

        if (tab === 'documents') loadDocuments();
        if (tab === 'qa') loadQAPairs();
        if (tab === 'system') { refreshHealth(); refreshMetrics(); }
    });
});

// ── Query Submission ───────────────────────────────────────
const queryInput = document.getElementById('query-input');
const btnAsk = document.getElementById('btn-ask');
const responseArea = document.getElementById('response-area');
const loadingSkeleton = document.getElementById('loading-skeleton');

btnAsk.addEventListener('click', handleAsk);
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAsk(); }
});

// ── Compare Submission ─────────────────────────────────────
const btnCompare = document.getElementById('btn-compare');
if (btnCompare) {
    btnCompare.addEventListener('click', handleCompare);
}

async function handleCompare() {
    const qInput = document.getElementById('compare-q-input').value.trim();
    const aInput = document.getElementById('compare-a-input').value.trim();
    
    if (!qInput || !aInput) {
        showToast('Please enter both a question and an answer to compare.', 'error');
        return;
    }
    
    const loading = document.getElementById('compare-loading');
    const resultArea = document.getElementById('compare-result-area');
    const btn = document.getElementById('btn-compare');
    
    loading.classList.remove('hidden');
    resultArea.classList.add('hidden');
    btn.disabled = true;
    
    try {
        const data = await API.compare(qInput, aInput);
        
        const verdictBox = document.getElementById('compare-verdict-box');
        const decisionText = document.getElementById('compare-decision');
        const iconContainer = document.getElementById('compare-icon');
        const sourcesSection = document.getElementById('compare-sources-section');
        const sourcesList = document.getElementById('compare-sources-list');
        
        verdictBox.className = 'compare-verdict'; // reset
        
        if (data.match === 'YES') {
            verdictBox.classList.add('verdict-yes');
            decisionText.textContent = 'YES';
            iconContainer.innerHTML = `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
            
            // Render sources if available
            if (data.sources && data.sources.length > 0) {
                sourcesList.innerHTML = '';
                data.sources.forEach(src => {
                    const item = document.createElement('div');
                    item.className = 'source-item';
                    const file = src.source_file || src.topic || 'Document';
                    const score = (src.similarity * 100).toFixed(1);
                    
                    const div = document.createElement('div');
                    div.innerText = src.content_preview || '';
                    const escapedContent = div.innerHTML;
                    
                    item.innerHTML = `
                        <div class="source-header">
                            <span class="source-file">${file}</span>
                            <span class="source-score">${score}% match</span>
                        </div>
                        <div class="source-preview">${escapedContent}</div>
                    `;
                    item.addEventListener('click', () => item.classList.toggle('expanded'));
                    sourcesList.appendChild(item);
                });
                sourcesSection.classList.remove('hidden');
            } else {
                sourcesSection.classList.add('hidden');
            }
            
        } else {
            verdictBox.classList.add('verdict-no');
            decisionText.textContent = 'NO';
            iconContainer.innerHTML = `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;
            sourcesSection.classList.add('hidden');
        }
        
        loading.classList.add('hidden');
        resultArea.classList.remove('hidden');
        showToast('Comparison complete', 'success');
        
    } catch (err) {
        loading.classList.add('hidden');
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
    }
}

async function handleAsk() {
    const text = queryInput.value.trim();

    // Check if we have a file selected instead
    if (state.selectedFile && state.selectedFileType) {
        return handleFileSubmit();
    }

    if (!text) { showToast('Please enter a question', 'error'); return; }

    const lang = document.getElementById('lang-select').value;
    const subject = document.getElementById('subject-select').value;
    const topic = document.getElementById('topic-select').value;

    showLoading();
    // Do not unhide responseArea yet
    const answerEl = document.getElementById('answer-text');
    answerEl.textContent = '';
    
    document.getElementById('sources-section').classList.add('hidden');
    document.getElementById('resp-cached').classList.add('hidden');
    document.getElementById('resp-latency').textContent = `- ms`;
    document.getElementById('resp-model').textContent = `-`;
    animateConfidenceBar('conf-answer', 0);
    animateConfidenceBar('conf-retrieval', 0);
    
    btnAsk.disabled = true;

    try {
        const resp = await fetch('/api/v1/ask/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, language: lang || 'en', subject, topic }),
        });

        if (!resp.ok) {
            const errText = await resp.text();
            try { throw new Error(JSON.parse(errText).detail || resp.statusText); } 
            catch(e) { throw new Error(errText || resp.statusText); }
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let done = false;

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
                const chunkStr = decoder.decode(value, { stream: true });
                const lines = chunkStr.split('\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6).trim();
                        if (dataStr === '[DONE]') {
                            break;
                        }
                        try {
                            const data = JSON.parse(dataStr);
                            if (data.metadata) {
                                const m = data.metadata;
                                if (m.latency_ms !== undefined) document.getElementById('resp-latency').textContent = `${m.latency_ms}ms`;
                                if (m.model_used) document.getElementById('resp-model').textContent = m.model_used;
                                if (m.cached) {
                                    document.getElementById('resp-cached').classList.remove('hidden');
                                } else {
                                    document.getElementById('resp-cached').classList.add('hidden');
                                }
                                if (m.confidence !== undefined) animateConfidenceBar('conf-answer', Math.round(m.confidence * 100));
                                if (m.retrieval_confidence !== undefined) animateConfidenceBar('conf-retrieval', Math.round(m.retrieval_confidence * 100));
                                
                                if (m.sources && m.sources.length > 0) {
                                    window._pendingSources = m.sources; // Store temporarily
                                }
                            }
                            if (data.chunk) {
                                if (responseArea.classList.contains('hidden')) {
                                    hideLoading();
                                    responseArea.classList.remove('hidden');
                                }
                                answerEl.textContent += data.chunk;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e, dataStr);
                        }
                    }
                }
            }
        }
        
        if (responseArea.classList.contains('hidden')) {
            hideLoading();
            responseArea.classList.remove('hidden');
        }
        
        // Stream completed, now render sources
        if (window._pendingSources) {
            const sourcesSection = document.getElementById('sources-section');
            const sourcesList = document.getElementById('sources-list');
            sourcesList.innerHTML = '';
            sourcesSection.classList.remove('hidden');
            window._pendingSources.forEach(src => {
                const item = document.createElement('div');
                item.className = 'source-item';
                const file = src.source_file || src.topic || 'Document';
                const score = (src.similarity * 100).toFixed(1);
                
                const div = document.createElement('div');
                div.innerText = src.content_preview || '';
                const escapedContent = div.innerHTML;
                
                item.innerHTML = `
                    <div class="source-header">
                        <span class="source-file">${file}</span>
                        <span class="source-score">${score}% match</span>
                    </div>
                    <div class="source-preview">${escapedContent}</div>
                `;
                item.addEventListener('click', () => item.classList.toggle('expanded'));
                sourcesList.appendChild(item);
            });
            window._pendingSources = null;
        }
        showToast('Answer generated!', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btnAsk.disabled = false;
    }
}

async function handleFileSubmit() {
    const lang = document.getElementById('lang-select').value;
    const subject = document.getElementById('subject-select').value;
    const topic = document.getElementById('topic-select').value;

    showLoading();
    try {
        let data;
        if (state.selectedFileType === 'audio') {
            data = await API.askAudio(state.selectedFile, lang, subject, topic);
            showResponse(data);
            showToast('Audio processed!', 'success');
        } else if (state.selectedFileType === 'video') {
            data = await API.askVideo(state.selectedFile, lang, subject, topic);
            showResponse(data);
            showToast('Video processed!', 'success');
        } else if (state.selectedFileType === 'ingest') {
            if (Array.isArray(state.selectedFile)) {
                showToast(`Started ingesting ${state.selectedFile.length} files in the background...`, 'info');
                for (const f of state.selectedFile) {
                    const res = await API.ingestFile(f, subject, topic, lang);
                    pollIngestStatus(res.document_id, f.name);
                }
                hideLoading();
            } else {
                showToast(`Started ingesting ${state.selectedFile.name}...`, 'info');
                const res = await API.ingestFile(state.selectedFile, subject, topic, lang);
                pollIngestStatus(res.document_id, state.selectedFile.name);
                hideLoading();
            }
        }
        clearFileSelection();
    } catch (err) {
        hideLoading();
        showToast(err.message, 'error');
    }
}

async function pollIngestStatus(documentId, filename) {
    const maxAttempts = 30; // 30 seconds max wait
    for (let i = 0; i < maxAttempts; i++) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        try {
            const statusData = await API.checkIngestStatus(documentId);
            if (statusData.status === 'completed') {
                showToast(`✅ ${filename} fully ingested! ${statusData.chunks_created} chunks created.`, 'success');
                return;
            } else if (statusData.status === 'failed') {
                showToast(`❌ Failed to ingest ${filename}.`, 'error');
                return;
            }
        } catch (e) {
            console.error(e);
        }
    }
    showToast(`⚠️ ${filename} is taking a long time to process. It will finish in the background.`, 'info');
}

function showLoading() {
    responseArea.classList.add('hidden');
    loadingSkeleton.classList.remove('hidden');
    btnAsk.disabled = true;
}

function hideLoading() {
    loadingSkeleton.classList.add('hidden');
    btnAsk.disabled = false;
}

function showResponse(data) {
    hideLoading();
    responseArea.classList.remove('hidden');

    // Answer text with typing effect
    const answerEl = document.getElementById('answer-text');
    answerEl.textContent = '';
    typeText(answerEl, data.answer || 'No answer received.');

    // Confidence bars
    const answerConf = Math.round((data.confidence || 0) * 100);
    const retrievalConf = Math.round((data.retrieval_confidence || 0) * 100);

    animateConfidenceBar('conf-answer', answerConf);
    animateConfidenceBar('conf-retrieval', retrievalConf);

    // Meta badges
    const cachedBadge = document.getElementById('resp-cached');
    if (data.cached) cachedBadge.classList.remove('hidden');
    else cachedBadge.classList.add('hidden');

    document.getElementById('resp-latency').textContent = `${data.latency_ms || 0}ms`;
    document.getElementById('resp-model').textContent = data.model_used || 'unknown';

    // Sources
    const sourcesSection = document.getElementById('sources-section');
    const sourcesList = document.getElementById('sources-list');
    sourcesList.innerHTML = '';

    if (data.sources && data.sources.length > 0) {
        sourcesSection.classList.remove('hidden');
        data.sources.forEach(src => {
            const item = document.createElement('div');
            item.className = 'source-item';
            item.innerHTML = `
                <div class="source-header">
                    <span class="source-file">${src.source_file || src.topic || 'Document'}</span>
                    <span class="source-score">${(src.similarity * 100).toFixed(1)}% match</span>
                </div>
                <div class="source-preview">${escapeHtml(src.content_preview || '')}</div>
            `;
            item.addEventListener('click', () => item.classList.toggle('expanded'));
            sourcesList.appendChild(item);
        });
    } else {
        sourcesSection.classList.add('hidden');
    }
}

function typeText(element, text, speed = 8) {
    let i = 0;
    const interval = setInterval(() => {
        if (i < text.length) {
            element.textContent += text[i];
            i++;
        } else {
            clearInterval(interval);
        }
    }, speed);
}

function animateConfidenceBar(prefix, percent) {
    const bar = document.getElementById(`${prefix}-bar`);
    const val = document.getElementById(`${prefix}-val`);

    bar.className = 'confidence-bar';
    if (percent >= 80) bar.classList.add('high');
    else if (percent >= 50) bar.classList.add('medium');
    else bar.classList.add('low');

    // Animate after a short delay
    setTimeout(() => {
        bar.style.width = `${percent}%`;
        val.textContent = `${percent}%`;
    }, 100);
}

// ── File Upload Handling ───────────────────────────────────
function setupDropZone(zoneId, inputId, fileType) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('keydown', (e) => { if (e.key === 'Enter') input.click(); });

    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            if (fileType === 'ingest') {
                selectFile(Array.from(e.dataTransfer.files), fileType);
            } else {
                selectFile(e.dataTransfer.files[0], fileType);
            }
        }
    });

    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            if (fileType === 'ingest') {
                selectFile(Array.from(input.files), fileType);
            } else {
                selectFile(input.files[0], fileType);
            }
        }
    });
}

setupDropZone('audio-drop', 'audio-file', 'audio');
setupDropZone('video-drop', 'video-file', 'video');
setupDropZone('file-drop', 'ingest-file', 'ingest');

function selectFile(file, type) {
    state.selectedFile = file;
    state.selectedFileType = type;
    const preview = document.getElementById('file-preview');
    const name = document.getElementById('file-preview-name');
    
    if (Array.isArray(file)) {
        name.textContent = `📎 ${file.length} files selected — ${type}`;
    } else {
        name.textContent = `📎 ${file.name} (${(file.size / 1024).toFixed(1)} KB) — ${type}`;
    }
    
    preview.classList.remove('hidden');
    btnAsk.textContent = type === 'ingest' ? '📄 Ingest File(s)' : '🎤 Process & Ask';
}

function clearFileSelection() {
    state.selectedFile = null;
    state.selectedFileType = null;
    document.getElementById('file-preview').classList.add('hidden');
    btnAsk.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg> Ask`;
    // Reset file inputs
    document.getElementById('audio-file').value = '';
    document.getElementById('video-file').value = '';
    document.getElementById('ingest-file').value = '';
}

document.getElementById('file-preview-clear').addEventListener('click', clearFileSelection);

// ── Audio Recording ────────────────────────────────────────
const btnRecord = document.getElementById('btn-record');
btnRecord.addEventListener('click', toggleRecording);

let audioSocket = null;

async function toggleRecording() {
    if (state.isRecording) {
        // Stop recording
        if (state.audioProcessor) {
            state.audioProcessor.disconnect();
            state.audioProcessor = null;
        }
        if (state.audioContext) {
            state.audioContext.close();
            state.audioContext = null;
        }
        if (state.mediaStream) {
            state.mediaStream.getTracks().forEach(track => track.stop());
            state.mediaStream = null;
        }
        if (audioSocket) audioSocket.close();
        
        state.isRecording = false;
        btnRecord.classList.remove('recording');
        btnRecord.querySelector('span').textContent = 'Record';
    } else {
        // Start recording via WebSocket (sending raw PCM for AssemblyAI)
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    autoGainControl: true,
                    echoCancellation: true,
                    noiseSuppression: false
                } 
            });
            
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(stream);
            // Use ScriptProcessorNode for wide browser compatibility to extract raw PCM
            const processor = audioContext.createScriptProcessor(4096, 1, 1);

            // Ensure previous connection is closed
            if (audioSocket) audioSocket.close();

            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            audioSocket = new WebSocket(`${wsProtocol}//${window.location.host}/api/v1/stream/dictation`);

            audioSocket.onopen = () => {
                state.isRecording = true;
                btnRecord.classList.add('recording');
                btnRecord.querySelector('span').textContent = 'Stop';
                showToast('Listening...', 'info');

                // Connect nodes to start processing
                source.connect(processor);
                processor.connect(audioContext.destination); // Required to make it process
            };

            processor.onaudioprocess = (e) => {
                if (audioSocket.readyState === WebSocket.OPEN) {
                    const inputData = e.inputBuffer.getChannelData(0);
                    // Convert float32 to int16 (PCM)
                    const pcmData = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        let s = Math.max(-1, Math.min(1, inputData[i]));
                        pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                    audioSocket.send(pcmData.buffer);
                }
            };

            state.audioContext = audioContext;
            state.audioProcessor = processor;
            state.mediaStream = stream;

            let finalTranscript = queryInput.value;
            if (finalTranscript && !finalTranscript.endsWith(' ')) {
                finalTranscript += ' ';
            }

            audioSocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'transcript') {
                    const text = data.text;
                    if (data.is_final) {
                        finalTranscript += text + ' ';
                        queryInput.value = finalTranscript;
                    } else {
                        queryInput.value = finalTranscript + text;
                    }
                } else if (data.type === 'error') {
                    showToast(data.message, 'error');
                    toggleRecording(); // Stop
                }
            };

            audioSocket.onerror = (error) => {
                console.error("WebSocket Error:", error);
                showToast('Connection to transcription server failed', 'error');
            };

            state.mediaRecorder.onstop = () => {
                stream.getTracks().forEach(track => track.stop());
            };

        } catch (err) {
            console.error("Recording error:", err);
            showToast('Microphone access denied or error starting recording.', 'error');
        }
    }
}

// ── Documents Management ───────────────────────────────────
async function loadDocuments() {
    const list = document.getElementById('documents-list');
    try {
        const data = await API.listDocuments();
        const docs = data.documents;
        if (!docs || docs.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>
                    <p>No documents uploaded yet. Go to the Ingest tab to add some.</p>
                </div>`;
            return;
        }

        list.innerHTML = docs.map(doc => {
            let statusBadge = '';
            if (doc.status === 'completed') statusBadge = '<span class="badge badge-success">Completed</span>';
            else if (doc.status === 'processing') statusBadge = '<span class="badge badge-warning">Processing...</span>';
            else statusBadge = '<span class="badge badge-error">Failed</span>';

            const date = new Date(doc.created_at).toLocaleString();

            return `
            <div class="qa-item" style="display:flex; justify-content: space-between; align-items: center; padding: 12px 16px;">
                <div style="display:flex; flex-direction: column; gap: 4px;">
                    <div style="font-weight: 500; font-size: 14px;">${escapeHtml(doc.filename)}</div>
                    <div style="font-size: 12px; color: var(--text-muted);">${date}</div>
                </div>
                <div style="display:flex; align-items: center; gap: 16px;">
                    <div style="font-size: 13px;">
                        <strong>${doc.chunks_created || 0}</strong> chunks
                    </div>
                    ${statusBadge}
                    <a href="/api/v1/ingest/documents/${doc.id}/view" target="_blank" class="btn btn-ghost btn-sm" title="View Document">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                    </a>
                    <button class="btn btn-ghost btn-sm btn-icon doc-delete-btn" title="Delete Document" data-doc-id="${doc.id}" data-doc-name="${escapeHtml(doc.filename)}">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                    </button>
                </div>
            </div>`;
        }).join('');

    } catch (err) {
        showToast('Failed to load documents: ' + err.message, 'error');
    }
}

document.getElementById('btn-refresh-docs')?.addEventListener('click', () => {
    loadDocuments();
    showToast('Documents refreshed', 'info');
});

// Event delegation for document delete buttons (avoids inline onclick XSS)
document.getElementById('documents-list')?.addEventListener('click', async (e) => {
    const btn = e.target.closest('.doc-delete-btn');
    if (!btn) return;
    const id = btn.dataset.docId;
    const filename = btn.dataset.docName;
    if (confirm(`Are you sure you want to completely delete "${filename}"?\nThis will remove the file and all its data chunks from the system.`)) {
        try {
            await API.deleteDocument(id);
            showToast(`Document deleted`, 'success');
            loadDocuments();
        } catch (err) {
            showToast('Delete failed: ' + err.message, 'error');
        }
    }
});

// ── Q/A Pairs Management ───────────────────────────────────
async function loadQAPairs() {
    const list = document.getElementById('qa-list');
    try {
        const pairs = await API.listQA(state.qaPage, state.qaPerPage);
        if (!pairs || pairs.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    <p>No Q/A pairs yet. Add one above or upload a file.</p>
                </div>`;
            return;
        }

        list.innerHTML = pairs.map(pair => `
            <div class="qa-item" data-id="${pair.id}">
                <div class="qa-item-content">
                    <div class="qa-item-display">
                        <div class="qa-item-q">${escapeHtml(pair.question)}</div>
                        <div class="qa-item-a">${escapeHtml(pair.answer)}</div>
                        <div class="qa-item-meta">
                            <span class="badge badge-muted">${pair.language || 'en'}</span>
                            <span class="badge badge-muted">v${pair.version || 1}</span>
                        </div>
                    </div>
                    <div class="qa-edit-inputs">
                        <input type="text" class="qa-edit-q" value="${escapeHtml(pair.question)}">
                        <textarea class="qa-edit-a" rows="2">${escapeHtml(pair.answer)}</textarea>
                        <div class="qa-edit-actions">
                            <button class="btn btn-primary btn-sm" onclick="saveQAEdit('${pair.id}', this)">Save</button>
                            <button class="btn btn-ghost btn-sm" onclick="cancelQAEdit(this)">Cancel</button>
                        </div>
                    </div>
                </div>
                <div class="qa-item-actions">
                    <button class="btn-icon" title="Edit" onclick="startQAEdit(this)">✏️</button>
                    <button class="btn-icon" title="Delete" onclick="confirmDeleteQA('${pair.id}')">🗑️</button>
                </div>
            </div>
        `).join('');

        document.getElementById('qa-page-info').textContent = `Page ${state.qaPage}`;
        document.getElementById('qa-prev').disabled = state.qaPage <= 1;
        document.getElementById('qa-next').disabled = pairs.length < state.qaPerPage;

    } catch (err) {
        showToast('Failed to load Q/A pairs: ' + err.message, 'error');
    }
}

// Inline edit functions (global for onclick)
window.startQAEdit = function(btn) {
    const item = btn.closest('.qa-item');
    item.classList.add('editing');
};

window.cancelQAEdit = function(btn) {
    const item = btn.closest('.qa-item');
    item.classList.remove('editing');
};

window.saveQAEdit = async function(id, btn) {
    const item = btn.closest('.qa-item');
    const question = item.querySelector('.qa-edit-q').value.trim();
    const answer = item.querySelector('.qa-edit-a').value.trim();
    if (!question || !answer) { showToast('Question and answer are required', 'error'); return; }
    try {
        await API.updateQA(id, question, answer, 'Edited via UI');
        showToast('Q/A pair updated!', 'success');
        loadQAPairs();
    } catch (err) {
        showToast('Update failed: ' + err.message, 'error');
    }
};

window.confirmDeleteQA = function(id) {
    state.deleteTargetId = id;
    document.getElementById('delete-modal').classList.remove('hidden');
};

// Delete modal
document.getElementById('modal-cancel').addEventListener('click', () => {
    document.getElementById('delete-modal').classList.add('hidden');
    state.deleteTargetId = null;
});
document.getElementById('modal-confirm').addEventListener('click', async () => {
    if (!state.deleteTargetId) return;
    try {
        await API.deleteQA(state.deleteTargetId);
        showToast('Q/A pair deleted', 'success');
        loadQAPairs();
    } catch (err) {
        showToast('Delete failed: ' + err.message, 'error');
    }
    document.getElementById('delete-modal').classList.add('hidden');
    state.deleteTargetId = null;
});
document.querySelector('.modal-backdrop')?.addEventListener('click', () => {
    document.getElementById('delete-modal').classList.add('hidden');
});

// Add Q/A pair
document.getElementById('btn-add-qa').addEventListener('click', async () => {
    const q = document.getElementById('qa-question').value.trim();
    const a = document.getElementById('qa-answer').value.trim();
    const lang = document.getElementById('qa-language').value;
    if (!q || !a) { showToast('Both question and answer are required', 'error'); return; }
    try {
        await API.createQA(q, a, lang);
        showToast('Q/A pair created!', 'success');
        document.getElementById('qa-question').value = '';
        document.getElementById('qa-answer').value = '';
        loadQAPairs();
    } catch (err) {
        showToast('Failed to create: ' + err.message, 'error');
    }
});

// Pagination
document.getElementById('qa-prev').addEventListener('click', () => {
    if (state.qaPage > 1) { state.qaPage--; loadQAPairs(); }
});
document.getElementById('qa-next').addEventListener('click', () => {
    state.qaPage++;
    loadQAPairs();
});

// Search (client-side filter)
document.getElementById('qa-search').addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    document.querySelectorAll('.qa-item').forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(query) ? '' : 'none';
    });
});

// ── System Health & Metrics ────────────────────────────────
async function refreshHealth() {
    try {
        const data = await API.getHealth();

        updateHealthItem('db', data.database);
        updateHealthItem('redis', data.redis);
        updateHealthItem('llm', data.llm);
        updateHealthItem('gpu', data.gpu_available ? 'available' : 'not available');

        // Update header dots
        updateStatusDot('status-db', data.database);
        updateStatusDot('status-redis', data.redis);
        updateStatusDot('status-llm', data.llm);
        document.getElementById('status-gpu').className =
            `status-dot ${data.gpu_available ? 'healthy' : 'degraded'}`;
    } catch (err) {
        ['db', 'redis', 'llm', 'gpu'].forEach(id => {
            const icon = document.getElementById(`health-${id}-icon`);
            const status = document.getElementById(`health-${id}-status`);
            if (icon) icon.className = 'health-icon unhealthy';
            if (status) status.textContent = 'Unreachable';
        });
    }
}

function updateHealthItem(id, statusText) {
    const icon = document.getElementById(`health-${id}-icon`);
    const status = document.getElementById(`health-${id}-status`);
    if (!icon || !status) return;

    const text = String(statusText || '');
    status.textContent = text;

    if (text.includes('healthy') || text === 'available') {
        icon.className = 'health-icon healthy';
    } else if (text.includes('unavailable') || text.includes('unhealthy') || text === 'not available') {
        icon.className = 'health-icon unhealthy';
    } else {
        icon.className = 'health-icon degraded';
    }
}

function updateStatusDot(dotId, statusText) {
    const dot = document.getElementById(dotId);
    if (!dot) return;
    const text = String(statusText || '');
    if (text.includes('healthy')) dot.className = 'status-dot healthy';
    else if (text.includes('unavailable') || text.includes('unhealthy')) dot.className = 'status-dot unhealthy';
    else dot.className = 'status-dot degraded';
}

async function refreshMetrics() {
    try {
        const data = await API.getMetrics();
        if (data.queue) {
            document.getElementById('metric-active').textContent = data.queue.active_requests || 0;
            document.getElementById('metric-queued').textContent = data.queue.queue_depth || 0;
            document.getElementById('metric-processed').textContent = data.queue.total_processed || 0;
        }
        if (data.cost) {
            const cost = data.cost.daily_estimated_cost_usd || 0;
            document.getElementById('metric-cost').textContent = `$${cost.toFixed(4)}`;
        }
    } catch (err) {
        // Metrics unavailable — leave current values
    }
}

document.getElementById('btn-refresh-health').addEventListener('click', () => {
    refreshHealth();
    refreshMetrics();
    showToast('Health refreshed', 'info');
});

// ── Utility ────────────────────────────────────────────────
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Auto-refresh ───────────────────────────────────────────
refreshHealth();
setInterval(refreshHealth, 30000);  // Every 30s
