document.addEventListener('DOMContentLoaded', function() {
  const chatForm = document.getElementById('chat-form');
  const messageInput = document.getElementById('message-input');
  const newSessionBtn = document.getElementById('new-session');
  const sessionLabel = document.getElementById('session-label');
  const typingIndicator = document.getElementById('typing-indicator');
  const chatContainer = document.querySelector('.chat-container');

  // Session handling
  function genSessionId(){return 's_' + Math.random().toString(36).slice(2,10);}  
  let sessionId = localStorage.getItem('chat_session_id');
  if(!sessionId){ sessionId = genSessionId(); localStorage.setItem('chat_session_id', sessionId); }
  renderSession();

  function renderSession(){ if(sessionLabel) sessionLabel.textContent = `#${sessionId.slice(-4)}`; }
  if(newSessionBtn){
    newSessionBtn.addEventListener('click', ()=>{
      sessionId = genSessionId();
      localStorage.setItem('chat_session_id', sessionId);
      // Clear UI messages
      [...chatContainer.querySelectorAll('.message-animation')].forEach(n=>n.remove());
      renderSession();
      addSystem('Bắt đầu phiên mới');
    });
  }

  messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
  });

  // No file upload UI in simplified version

  async function sendMessage(message) {
    if (!message) return;
    addMessage(message, 'user');

    typingIndicator.classList.remove('hidden');
    chatContainer.scrollTop = chatContainer.scrollHeight;

    messageInput.value = '';
    messageInput.style.height = 'auto';
  // build SSE url with session

  const sseUrl = '/invoke/stream?q=' + encodeURIComponent(message || '') + '&session_id=' + encodeURIComponent(sessionId);
    if (window.EventSource) {
      let aiContainer = null;
      let outEl = null;
      let receivedAny = false;
      let buffer = '';
      let typingInterval = null;
      let doneSignal = false;
      const typingSpeedCPS = 35;
      const intervalMs = 30;
      const charsPerTick = Math.max(1, Math.round(typingSpeedCPS * intervalMs / 1000));

      function createAIContainer(){
        if (aiContainer) return;
        aiContainer = document.createElement('div');
        aiContainer.className = 'message-animation max-w-3xl mx-auto';
        aiContainer.innerHTML = `
          <div class="flex space-x-3">
            <div class="flex-shrink-0">
              <div class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                <i class="fas fa-robot"></i>
              </div>
            </div>
            <div class="bg-white p-4 rounded-lg shadow-sm">
              <p class="text-gray-800 whitespace-pre-wrap" id="_sse_output" style="display:none"></p>
            </div>
          </div>`;
        chatContainer.insertBefore(aiContainer, typingIndicator);
        outEl = aiContainer.querySelector('#_sse_output');
      }

      function startTypingLoop(){
        if (typingInterval) return;
        typingInterval = setInterval(()=>{
          if (buffer.length === 0){
            if (doneSignal){
              clearInterval(typingInterval);
              typingInterval = null;
              typingIndicator.classList.add('hidden');
            }
            return;
          }
          const chunk = buffer.slice(0, charsPerTick);
          buffer = buffer.slice(chunk.length);
          if (outEl){
            if (outEl.style.display === 'none') outEl.style.display = 'block';
            outEl.textContent += chunk;
          }
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }, intervalMs);
      }

      let es; try { es = new EventSource(sseUrl); } catch(e){ es = null; }
      if (es){
        es.onmessage = function(ev){
          if (ev.data === '[DONE]') return;
          if (!aiContainer) createAIContainer();
          if (!receivedAny) typingIndicator.classList.add('hidden');
          receivedAny = true;
            buffer += ev.data;
          startTypingLoop();
        };
        es.addEventListener('done', () => { es.close(); doneSignal = true; });
        es.onerror = function(){ es.close(); doneSignal = true; if (!receivedAny) fallbackFetch(null,false); };
      } else fallbackFetch(null,false);
    } else fallbackFetch(null,false);

  async function fallbackFetch(outEl, inline){
      try {
        const resp = await fetch('/invoke', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: message || '', session_id: sessionId })
        });
        if (!resp.ok){
          const text = await resp.text();
          if (inline && outEl) outEl.textContent = 'Lỗi server: ' + resp.status + ' ' + text; else addMessage('Lỗi server: ' + resp.status + ' ' + text, 'ai', 'text');
        } else {
          const data = await resp.json();
          const out = (data && (data.output || data.result || data.message)) ? (data.output || data.result || data.message) : JSON.stringify(data);
          if (inline && outEl) outEl.textContent = out; else addMessage(out, 'ai', 'text');
        }
      } catch(err){
        if (inline && outEl) outEl.textContent = 'Lỗi kết nối: ' + err.message; else addMessage('Lỗi kết nối: ' + err.message, 'ai', 'text');
      } finally {
        typingIndicator.classList.add('hidden');
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    }
  }

  chatForm.addEventListener('submit', function(e) {
    e.preventDefault();
    const message = messageInput.value.trim();
    sendMessage(message);
  });

  messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const message = messageInput.value.trim();
      sendMessage(message);
    }
  });

  function addMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message-animation max-w-3xl mx-auto';

    if (sender === 'user') {
      messageDiv.innerHTML = `
        <div class="flex space-x-3 justify-end">
          <div class="bg-blue-500 text-white p-3 rounded-lg shadow-sm max-w-[70%]">
            <p>${content}</p>
          </div>
          <div class="flex-shrink-0">
            <div class="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center text-gray-600">
              <i class="fas fa-user"></i>
            </div>
          </div>
        </div>`;
    } else {
      messageDiv.innerHTML = `
        <div class="flex space-x-3">
          <div class="flex-shrink-0">
            <div class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
              <i class="fas fa-robot"></i>
            </div>
          </div>
          <div class="bg-gray-100 p-3 rounded-lg shadow-sm max-w-[70%]">
            <p class="text-gray-800">${content}</p>
          </div>
        </div>`;
    }

    chatContainer.insertBefore(messageDiv, typingIndicator);
  }

  function addSystem(content){
    const div = document.createElement('div');
    div.className = 'text-center text-[11px] text-gray-400';
    div.textContent = content;
    chatContainer.insertBefore(div, typingIndicator);
  }
});
