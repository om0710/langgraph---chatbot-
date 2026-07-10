document.addEventListener("DOMContentLoaded", () => {
    const chatMessages = document.getElementById("chat-messages");
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const newChatBtn = document.getElementById("new-chat-btn");
    const threadsList = document.getElementById("threads-list");
    const activeThreadIdLabel = document.getElementById("active-thread-id");
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const uploadStatus = document.getElementById("upload-status");

    let currentThreadId = localStorage.getItem("currentThreadId") || uuidv4();
    localStorage.setItem("currentThreadId", currentThreadId);

    // Initialize Page
    updateActiveThreadLabel();
    fetchThreads();
    loadThreadHistory(currentThreadId);

    // Generate standard UUID v4 in vanilla JS
    function uuidv4() {
        return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
            (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
        );
    }

    function updateActiveThreadLabel() {
        const shortId = currentThreadId.substring(0, 8) + "...";
        activeThreadIdLabel.textContent = `Session: ${shortId}`;
    }

    // Fetch list of active threads from SQLite
    async function fetchThreads() {
        try {
            const res = await fetch("/threads");
            const data = await res.json();
            threadsList.innerHTML = "";
            
            // Add current thread to sidebar if not present
            let threads = data.threads || [];
            if (!threads.includes(currentThreadId)) {
                threads.push(currentThreadId);
            }

            threads.forEach(thread => {
                const li = document.createElement("li");
                li.className = `thread-item ${thread === currentThreadId ? "active" : ""}`;
                const shortId = thread.substring(0, 8) + "...";
                li.innerHTML = `
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    <span>Chat ${shortId}</span>
                `;
                li.addEventListener("click", () => {
                    selectThread(thread);
                });
                threadsList.appendChild(li);
            });
        } catch (e) {
            console.error("Error loading threads:", e);
        }
    }

    // Switch active thread session
    function selectThread(threadId) {
        currentThreadId = threadId;
        localStorage.setItem("currentThreadId", currentThreadId);
        updateActiveThreadLabel();
        
        // Highlight active session
        document.querySelectorAll(".thread-item").forEach(item => {
            item.classList.remove("active");
        });
        fetchThreads();
        loadThreadHistory(currentThreadId);
    }

    // Load thread history
    async function loadThreadHistory(threadId) {
        try {
            const res = await fetch(`/thread/${threadId}/history`);
            const data = await res.json();
            
            chatMessages.innerHTML = "";
            if (!data.messages || data.messages.length === 0) {
                showWelcomeMessage();
                return;
            }

            data.messages.forEach(msg => {
                appendMessage(msg.role, msg.content);
            });
        } catch (e) {
            console.error("Error loading chat history:", e);
            showWelcomeMessage();
        }
    }

    function showWelcomeMessage() {
        chatMessages.innerHTML = `
            <div class="welcome-message">
                <h2>Hello! How can I assist you today?</h2>
                <p>I have access to your PDF knowledge base, mathematical tools, Wikipedia, and live stock queries.</p>
            </div>
        `;
    }

    function appendMessage(role, content) {
        // Remove welcome message if present
        const welcome = document.querySelector(".welcome-message");
        if (welcome) welcome.remove();

        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role}`;
        
        // Convert simple markdown code block formats
        let formattedContent = content
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/`([^`]+)`/g, "<code>$1</code>")
            .replace(/\n/g, "<br>");
            
        msgDiv.innerHTML = formattedContent;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return msgDiv;
    }

    // New Chat Button Click
    newChatBtn.addEventListener("click", () => {
        const newId = uuidv4();
        selectThread(newId);
    });

    // Send Message Form Submit
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        userInput.value = "";
        appendMessage("user", text);

        // Append assistant loading message container
        const assistantBubble = appendMessage("assistant", "");
        let fullResponse = "";

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: text, thread_id: currentThreadId })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            
            // Read streamed response
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split("\n");
                
                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        try {
                            const parsed = JSON.parse(line.substring(6));
                            if (parsed.text) {
                                fullResponse += parsed.text;
                                // Display typewriter streaming
                                assistantBubble.innerHTML = fullResponse
                                    .replace(/`([^`]+)`/g, "<code>$1</code>")
                                    .replace(/\n/g, "<br>") + "▌";
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            } else if (parsed.error) {
                                assistantBubble.innerHTML = `<span style="color:#ef4444;">Error: ${parsed.error}</span>`;
                            }
                        } catch (e) {
                            // JSON parsing skipped for partial chunk parts
                        }
                    }
                }
            }
            
            // Remove typewriter cursor once finished
            assistantBubble.innerHTML = fullResponse
                .replace(/`([^`]+)`/g, "<code>$1</code>")
                .replace(/\n/g, "<br>");
            
            // Refresh thread list sidebar to show the active chat session
            fetchThreads();

        } catch (err) {
            assistantBubble.innerHTML = `<span style="color:#ef4444;">Error connecting to server.</span>`;
            console.error(err);
        }
    });

    // Drag and Drop File Upload
    uploadZone.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });

    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = "#c084fc";
        uploadZone.style.backgroundColor = "rgba(124, 58, 237, 0.05)";
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.style.borderColor = "rgba(124, 58, 237, 0.3)";
        uploadZone.style.backgroundColor = "rgba(255, 255, 255, 0.01)";
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = "rgba(124, 58, 237, 0.3)";
        uploadZone.style.backgroundColor = "rgba(255, 255, 255, 0.01)";
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    async function uploadFile(file) {
        if (!file.name.endsWith(".pdf")) {
            showUploadStatus("❌ Only PDF files are supported.", "#ef4444");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        showUploadStatus("⏳ Uploading file...", "#c084fc");

        try {
            const res = await fetch("/upload", {
                method: "POST",
                body: formData
            });

            if (res.ok) {
                showUploadStatus("✅ Document context uploaded and indexed!", "#10b981");
                // Clear file input
                fileInput.value = "";
            } else {
                const data = await res.json();
                showUploadStatus(`❌ Error: ${data.detail || "Upload failed."}`, "#ef4444");
            }
        } catch (e) {
            showUploadStatus("❌ Network error uploading file.", "#ef4444");
        }
    }

    function showUploadStatus(text, color) {
        uploadStatus.textContent = text;
        uploadStatus.style.color = color;
    }
});
