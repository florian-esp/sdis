import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import logoSDIS from './assets/logo-sdids.png';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll vers le bas
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Gestion de l'upload de fichiers
  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    const newFiles = files.map(file => ({
      id: Date.now() + Math.random(),
      file: file,
      name: file.name,
      type: file.type,
      size: file.size,
      preview: null,
      uploadDate: new Date()
    }));

    // Créer des previews pour les images
    newFiles.forEach(fileObj => {
      if (fileObj.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onloadend = () => {
          fileObj.preview = reader.result;
          setUploadedFiles(prev => [...prev]);
        };
        reader.readAsDataURL(fileObj.file);
      }
    });

    setUploadedFiles(prev => [...prev, ...newFiles]);
  };

  // Supprimer un fichier uploadé
  const removeFile = (fileId) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
    if (selectedFile?.id === fileId) {
      setSelectedFile(null);
    }
  };

  // Formater la taille du fichier
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  // Obtenir l'icône selon le type de fichier
  const getFileIcon = (type) => {
    if (type.startsWith('image/')) return '📷';
    if (type.includes('pdf')) return '📄';
    if (type.includes('word') || type.includes('document')) return '📝';
    if (type.includes('sheet') || type.includes('excel')) return '📊';
    if (type.includes('video')) return '📹';
    if (type.includes('audio')) return '🔊';
    return '📎';
  };

  // Envoyer le message
  const handleSendMessage = async () => {
    if (!inputValue.trim() && uploadedFiles.length === 0) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: inputValue,
      files: [...uploadedFiles],
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Simuler une réponse de l'IA (à remplacer par votre API)
    setTimeout(() => {
      const botResponse = {
        id: Date.now() + 1,
        type: 'bot',
        text: generateBotResponse(userMessage),
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botResponse]);
      setIsLoading(false);
    }, 1500);
  };

  // Générer une réponse du bot
  const generateBotResponse = (userMessage) => {
    if (userMessage.files.length > 0) {
      const fileNames = userMessage.files.map(f => f.name).join(', ');
      return `J'ai bien reçu ${userMessage.files.length} document(s) : ${fileNames}. ${
        userMessage.text ? `Concernant votre demande "${userMessage.text}", j'analyse le contenu des fichiers transmis.` : 'Comment puis-je vous assister avec ces documents ?'
      }`;
    }
    return `Message reçu. Comment puis-je vous aider concernant : "${userMessage.text}" ?`;
  };

  // Gestion de la touche Entrée
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Visualiser un fichier
  const handleFileClick = (file) => {
    setSelectedFile(file);
  };

  return (
    <div className="app">
      {/* Barre latérale des fichiers */}
      <div className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <h3>Fichiers joints</h3>
          <button 
            className="sidebar-toggle"
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          >
            {isSidebarOpen ? '◀' : '▶'}
          </button>
        </div>
        
        {isSidebarOpen && (
          <div className="sidebar-content">
            {uploadedFiles.length === 0 ? (
              <div className="sidebar-empty">
                <p>Aucun fichier joint</p>
                <span className="sidebar-empty-icon">📁</span>
              </div>
            ) : (
              <div className="sidebar-files">
                {uploadedFiles.map(file => (
                  <div 
                    key={file.id} 
                    className={`sidebar-file-item ${selectedFile?.id === file.id ? 'selected' : ''}`}
                    onClick={() => handleFileClick(file)}
                  >
                    <div className="sidebar-file-icon">
                      {getFileIcon(file.type)}
                    </div>
                    <div className="sidebar-file-info">
                      <span className="sidebar-file-name">{file.name}</span>
                      <span className="sidebar-file-size">{formatFileSize(file.size)}</span>
                    </div>
                    <button 
                      className="sidebar-file-remove"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile(file.id);
                      }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Conteneur principal */}
      <div className="main-container">
        <div className="chat-container">
          {/* Header */}
          <div className="chat-header">
            <div className="header-content">
              <div className="header-logo">
                <img 
                  src={logoSDIS}
                  alt="Logo SDIS" 
                  className="sdis-logo"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'flex';
                  }}
                />
                <div className="logo-fallback">SDIS</div>
              </div>
              <div className="header-info">
                <h1>Assistant SDIS</h1>
                <p className="header-subtitle">Service Départemental d'Incendie et de Secours</p>
              </div>
            </div>
          </div>

          {/* Messages */}
          <div className="messages-container">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <div className="welcome-icon-container">
                  <svg className="welcome-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                </div>
                <h2>Bienvenue sur l'Assistant SDIS</h2>
                <p>Posez vos questions ou joignez des documents pour analyse.<br/>
                Utilisez le bouton de pièce jointe pour ajouter vos fichiers.</p>
              </div>
            ) : (
              messages.map(message => (
                <div key={message.id} className={`message ${message.type}`}>
                  <div className="message-avatar">
                    {message.type === 'bot' ? (
                      <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                      </svg>
                    )}
                  </div>
                  <div className="message-content">
                    {message.files && message.files.length > 0 && (
                      <div className="message-files">
                        <span className="files-badge">{message.files.length} fichier(s) joint(s)</span>
                      </div>
                    )}
                    {message.text && <p className="message-text">{message.text}</p>}
                    <span className="message-time">
                      {message.timestamp.toLocaleTimeString('fr-FR', { 
                        hour: '2-digit', 
                        minute: '2-digit' 
                      })}
                    </span>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message bot">
                <div className="message-avatar">
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                  </svg>
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Zone de saisie */}
          <div className="input-container">
            <div className="input-wrapper">
              <button 
                className="attach-btn"
                onClick={() => fileInputRef.current?.click()}
                title="Joindre un fichier"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                </svg>
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                multiple
                accept=".pdf,.jpg,.jpeg,.png,.gif,.docx,.doc,.xlsx,.xls,.txt,.mp4,.mp3"
                style={{ display: 'none' }}
              />
              <textarea
                className="message-input"
                placeholder="Saisissez votre message..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                rows="1"
              />
              <button 
                className="send-btn"
                onClick={handleSendMessage}
                disabled={!inputValue.trim() && uploadedFiles.length === 0}
              >
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Panneau de prévisualisation */}
        {selectedFile && (
          <div className="preview-panel">
            <div className="preview-header">
              <h3>Aperçu du fichier</h3>
              <button 
                className="preview-close"
                onClick={() => setSelectedFile(null)}
              >
                ✕
              </button>
            </div>
            <div className="preview-content">
              <div className="preview-file-info">
                <div className="preview-file-icon-large">
                  {getFileIcon(selectedFile.type)}
                </div>
                <h4>{selectedFile.name}</h4>
                <p className="preview-file-meta">
                  Taille: {formatFileSize(selectedFile.size)}<br/>
                  Type: {selectedFile.type || 'Inconnu'}
                </p>
              </div>
              
              {selectedFile.preview ? (
                <div className="preview-image-container">
                  <img src={selectedFile.preview} alt={selectedFile.name} />
                </div>
              ) : (
                <div className="preview-placeholder">
                  <p>Aperçu non disponible pour ce type de fichier</p>
                  <span className="preview-placeholder-icon">{getFileIcon(selectedFile.type)}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;