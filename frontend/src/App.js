// src/App.js
import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './App.css';

const USER1_ID = '12345678-1234-5678-1234-567812345678';
const USER2_ID = '87654321-4321-8765-4321-876543210987';

const API_BASE_URL = 'http://localhost:8000';
const WS_BASE_URL = 'ws://localhost:8000';

const USERS = {
  'User1': { id: USER1_ID, username: 'user1', name: 'User 1', avatar: '👤' },
  'User2': { id: USER2_ID, username: 'user2', name: 'User 2', avatar: '👥' }
};

function App() {
  const [user, setUser] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentView, setCurrentView] = useState('login');
  const [wsConnected, setWsConnected] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [showFileSelector, setShowFileSelector] = useState(false);
  const [uploading, setUploading] = useState(false);
  
  const wsRef = useRef(null);
  const messageQueueRef = useRef([]);

  const handleWebSocketMessage = useCallback((data) => {
    const { type } = data;
    
    if (type === 'answer') {
      const { conversation_id, answer } = data;
      if (conversation_id === currentConversation?.id) {
        setLoading(false);
        setMessages(prev => [...prev, { role: 'assistant', content: answer }]);
      }
    } else if (type === 'ack') {
      console.log('Message acknowledged');
    } else if (type === 'error') {
      console.error('Server error:', data.error);
      setLoading(false);
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${data.error}` }]);
    }
  }, [currentConversation?.id]);

  const connectWebSocket = useCallback(() => {
    if (!user) return;
    
    const wsUrl = `${WS_BASE_URL}/ws/${user.id}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setWsConnected(true);
      
      while (messageQueueRef.current.length > 0) {
        const msg = messageQueueRef.current.shift();
        ws.send(JSON.stringify(msg));
      }
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsConnected(false);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWsConnected(false);
      setTimeout(() => {
        if (user && currentView === 'chat') {
          connectWebSocket();
        }
      }, 3000);
    };
    
    wsRef.current = ws;
  }, [user, currentView, handleWebSocketMessage]);

  useEffect(() => {
    if (user && currentView === 'chat') {
      connectWebSocket();
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [user, currentView, connectWebSocket]);

  const sendWebSocketMessage = useCallback((conversationId, prompt, fileIds = []) => {
    const message = {
      type: 'chat',
      conversation_id: conversationId,
      prompt: prompt,
      file_ids: fileIds
    };
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      messageQueueRef.current.push(message);
    }
  }, []);

  const loadConversations = useCallback(async () => {
    if (!user) return;
    try {
      const response = await axios.get(`${API_BASE_URL}/conversation/user/${user.id}`);
      setConversations(response.data);
    } catch (error) {
      console.error('Error loading conversations:', error);
    }
  }, [user]);

  const loadUploadedFiles = useCallback(async () => {
    if (!user) return;
    try {
      const response = await axios.get(`${API_BASE_URL}/documents/`, {
        params: { skip: 0, limit: 100 }
      });
      const files = response.data.documents || response.data;
      setUploadedFiles(Array.isArray(files) ? files : []);
    } catch (error) {
      console.error('Error loading files:', error);
    }
  }, [user]);

  useEffect(() => {
    if (user) {
      loadConversations();
      loadUploadedFiles();
    }
  }, [user, loadConversations, loadUploadedFiles]);

  const handleLogin = (userKey) => {
    const selectedUser = USERS[userKey];
    setUser(selectedUser);
    setCurrentView('dashboard');
  };

  const handleLogout = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setUser(null);
    setConversations([]);
    setCurrentConversation(null);
    setMessages([]);
    setUploadedFiles([]);
    setSelectedFiles([]);
    setWsConnected(false);
    setCurrentView('login');
  };

  const createNewConversation = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/conversation/new`, {
        user_id: user.id
      });
      const newConversation = response.data;
      setCurrentConversation({ id: newConversation.conversation_id });
      setMessages([]);
      setSelectedFiles([]);
      // No automatic file selector popup - user must click "Docs" button
      setCurrentView('chat');
      await loadConversations();
    } catch (error) {
      console.error('Error creating conversation:', error);
    }
  };

  const loadConversation = async (conversationId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/conversation/${conversationId}`);
      const conversation = response.data;
      setCurrentConversation({ id: conversationId });
      
      const formattedMessages = [];
      conversation.dialogues.forEach(dialogue => {
        formattedMessages.push({ role: 'user', content: dialogue.prompt });
        if (dialogue.answer) {
          formattedMessages.push({ role: 'assistant', content: dialogue.answer });
        }
      });
      setMessages(formattedMessages);
      setShowFileSelector(false);
      setCurrentView('chat');
    } catch (error) {
      console.error('Error loading conversation:', error);
    }
  };

  const sendMessage = () => {
    if (!inputMessage.trim() || !currentConversation) return;

    const userMessage = inputMessage;
    setInputMessage('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);
    setShowFileSelector(false);

    const fileIds = selectedFiles.map(f => f.id);
    sendWebSocketMessage(currentConversation.id, userMessage, fileIds);
  };

  const deleteConversation = async (conversationId) => {
    try {
      await axios.delete(`${API_BASE_URL}/conversation/${conversationId}`);
      await loadConversations();
      if (currentConversation?.id === conversationId) {
        setCurrentConversation(null);
        setMessages([]);
        setCurrentView('dashboard');
      }
    } catch (error) {
      console.error('Error deleting conversation:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', user.id);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/documents/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      alert(`File "${response.data.name}" uploaded successfully!`);
      await loadUploadedFiles();
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file. Please try again.');
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const deleteFile = async (fileId, fileName) => {
    if (window.confirm(`Are you sure you want to delete "${fileName}"?`)) {
      try {
        await axios.delete(`${API_BASE_URL}/documents/${fileId}`);
        alert(`File "${fileName}" deleted successfully!`);
        await loadUploadedFiles();
        setSelectedFiles(prev => prev.filter(f => f.id !== fileId));
      } catch (error) {
        console.error('Error deleting file:', error);
        alert('Error deleting file. Please try again.');
      }
    }
  };

  const toggleFileSelection = (file) => {
    setSelectedFiles(prev => {
      if (prev.find(f => f.id === file.id)) {
        return prev.filter(f => f.id !== file.id);
      } else {
        return [...prev, file];
      }
    });
  };

  const getFileIcon = (fileName) => {
    const ext = fileName.split('.').pop().toLowerCase();
    switch(ext) {
      case 'pdf': return '📕';
      case 'docx': return '📘';
      case 'doc': return '📘';
      case 'txt': return '📄';
      case 'md': return '📝';
      default: return '📎';
    }
  };

  if (currentView === 'login') {
    return (
      <div className="login">
        <div className="login-card">
          <div className="login-icon">✨</div>
          <h1>RAG Chat</h1>
          <p>Intelligent document Q&A</p>
          <div className="login-users">
            <button onClick={() => handleLogin('User1')}>
              <span>👤</span> User 1
            </button>
            <button onClick={() => handleLogin('User2')}>
              <span>👥</span> User 2
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="user-info">
            <div className="user-avatar">{user.avatar}</div>
            <div>
              <div className="user-name">{user.name}</div>
              <div className="user-status">
                <span className={`status-dot ${wsConnected ? 'online' : 'offline'}`}></span>
                {wsConnected ? 'Online' : 'Offline'}
              </div>
            </div>
          </div>
          <button onClick={handleLogout} className="logout">Logout</button>
        </div>

        <button onClick={createNewConversation} className="new-chat-btn">
          <span>+</span> New Chat
        </button>

        <div className="conversation-list">
          <div className="section-title">Recent Chats</div>
          {conversations.length === 0 ? (
            <div className="empty">No chats yet</div>
          ) : (
            conversations.map(conv => (
              <div
                key={conv.id}
                className={`conversation-item ${currentConversation?.id === conv.id ? 'active' : ''}`}
                onClick={() => loadConversation(conv.id)}
              >
                <span>💬</span>
                <span className="conv-title">{conv.title || conv.id.slice(0, 8)}</span>
                <button className="delete-chat" onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id); }}>✕</button>
              </div>
            ))
          )}
        </div>
      </aside>

      <main className="chat-area">
        {currentView === 'dashboard' ? (
          <div className="dashboard">
            <div className="dashboard-content">
              <div className="dashboard-icon">📚</div>
              <h2>Welcome, {user.name}</h2>
              <p>Upload documents to get started</p>
              
              <div className="upload-section">
                <label className="upload-btn">
                  {uploading ? 'Uploading...' : '📄 Upload Document'}
                  <input 
                    type="file" 
                    accept=".pdf,.docx,.doc,.txt,.md" 
                    onChange={handleFileUpload} 
                    disabled={uploading}
                  />
                </label>
                
                {uploading && (
                  <div className="uploading-indicator">
                    <div className="spinner"></div>
                    <span>Uploading...</span>
                  </div>
                )}

                <div className="supported-formats">
                  <small>Supported: PDF, DOCX, DOC, TXT, MD</small>
                </div>
                
                {uploadedFiles.length > 0 && (
                  <div className="file-grid">
                    <h4>Your Documents ({uploadedFiles.length})</h4>
                    {uploadedFiles.map(file => (
                      <div key={file.id} className="file-card">
                        <span className="file-icon">{getFileIcon(file.name)}</span>
                        <div className="file-info">
                          <div className="file-name" title={file.name}>
                            {file.name.length > 30 ? file.name.substring(0, 30) + '...' : file.name}
                          </div>
                          <div className="file-date">
                            {new Date(file.created_at).toLocaleDateString()}
                          </div>
                        </div>
                        <button 
                          className="delete-file" 
                          onClick={() => deleteFile(file.id, file.name)}
                          title="Delete"
                        >
                          ✕
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <button className="start-chat" onClick={createNewConversation}>
                Start a new chat →
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="chat-header">
              <div className="chat-title">
                <span>💬</span> Chat
                {selectedFiles.length > 0 && (
                  <span className="doc-count-badge" title={`${selectedFiles.length} document(s) selected`}>
                    📄 {selectedFiles.length}
                  </span>
                )}
              </div>
              <div className="chat-header-buttons">
                <button 
                  onClick={() => setShowFileSelector(true)} 
                  className="docs-btn"
                  title="View uploaded documents"
                >
                  📄 Docs
                </button>
                <button onClick={() => setCurrentView('dashboard')} className="back-btn">
                  ← Dashboard
                </button>
              </div>
            </div>

            {showFileSelector && (
              <div className="file-modal">
                <div className="file-modal-content">
                  <div className="file-modal-header">
                    <h3>Your Documents</h3>
                    <button onClick={() => setShowFileSelector(false)}>✕</button>
                  </div>
                  <div className="file-modal-body">
                    {uploadedFiles.length === 0 ? (
                      <div className="no-docs-message">
                        <p>No documents uploaded yet.</p>
                        <p>Upload documents from the dashboard.</p>
                      </div>
                    ) : (
                      <>
                        <div className="file-list">
                          {uploadedFiles.map(file => (
                            <label key={file.id} className="file-item">
                              <input
                                type="checkbox"
                                checked={selectedFiles.some(f => f.id === file.id)}
                                onChange={() => toggleFileSelection(file)}
                              />
                              <span className="file-icon">{getFileIcon(file.name)}</span>
                              <span className="file-name">{file.name}</span>
                            </label>
                          ))}
                        </div>
                        <div className="file-modal-footer">
                          <span className="selected-info">
                            {selectedFiles.length} document{selectedFiles.length !== 1 ? 's' : ''} selected
                          </span>
                          <button onClick={() => setShowFileSelector(false)} className="confirm-btn">
                            Done
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}

            <div className="messages">
              {messages.length === 0 ? (
                <div className="empty-messages">
                  <div className="empty-icon">💬</div>
                  <h3>New conversation</h3>
                  <p>Ask a question to get started</p>
                  {selectedFiles.length > 0 && (
                    <div className="docs-badge">
                      📄 Using {selectedFiles.length} document{selectedFiles.length !== 1 ? 's' : ''}
                    </div>
                  )}
                </div>
              ) : (
                <>
                  {selectedFiles.length > 0 && messages.length > 0 && (
                    <div className="docs-badge inline">
                      📄 Using: {selectedFiles.map(f => f.name).join(', ')}
                    </div>
                  )}
                  {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                      <div className="message-avatar">{msg.role === 'user' ? user.avatar : '🤖'}</div>
                      <div className="message-text">{msg.content}</div>
                    </div>
                  ))}
                </>
              )}
              {loading && (
                <div className="message assistant">
                  <div className="message-avatar">🤖</div>
                  <div className="message-text typing">Thinking...</div>
                </div>
              )}
            </div>

            <div className="input-bar">
              <button onClick={() => setShowFileSelector(true)} className="attach" title="Select documents">📎</button>
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                placeholder={selectedFiles.length > 0 ? `Ask about ${selectedFiles.length} document(s)...` : "Ask a question..."}
                disabled={loading}
              />
              <button onClick={sendMessage} disabled={loading || !inputMessage.trim()}>Send</button>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;