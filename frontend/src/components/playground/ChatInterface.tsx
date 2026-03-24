import React, { useState, useRef, useEffect, useCallback } from 'react';
import { apiService } from '../../services/api';
import MessageContent from './MessageContent';
import VideoPlayer from './VideoPlayer';
import SearchFilters from './SearchFilters';
import type { SearchFilterMetadataField } from './SearchFilters';
import Modal from '../ui/Modal';

interface AIService {
  service_id: number;
  name: string;
}

interface VideoReference {
  file_id: string;
  filename: string;
  start_time: number;
  end_time: number;
  start_formatted: string;
  end_formatted: string;
  text_preview: string;
  video_url: string;
  is_agent_cited?: boolean;  // True if agent explicitly cited this timestamp
}

interface Message {
  id: string;
  type: 'user' | 'agent' | 'error';
  content: string;
  timestamp: Date;
  files?: string[];
  videoReferences?: VideoReference[];
}

interface ChatInterfaceProps {
  appId: number;
  agentId: number;
  agentName: string;
  conversationId?: number | null;
  onConversationCreated?: (conversationId: number) => void;
  onMessageSent?: () => void;
  metadataFields?: SearchFilterMetadataField[];
  vectorDbType?: string;
}

function ChatInterface({
  appId,
  agentId,
  agentName,
  conversationId,
  onConversationCreated,
  onMessageSent,
  metadataFields,
  vectorDbType,
}: Readonly<ChatInterfaceProps>) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [persistentFiles, setPersistentFiles] = useState<any[]>([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [isFilterExpanded, setIsFilterExpanded] = useState(false);
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(conversationId || null);
  const [filterMetadata, setFilterMetadata] = useState<Record<string, unknown> | undefined>(undefined);
  const [filtersKey, setFiltersKey] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const filterPanelId = `metadata-filters-${agentId}`;

  // Media upload state
  const [showMediaUploadModal, setShowMediaUploadModal] = useState(false);
  const [mediaFiles, setMediaFiles] = useState<File[]>([]);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [aiServices, setAiServices] = useState<AIService[]>([]);
  const [selectedTranscriptionServiceId, setSelectedTranscriptionServiceId] = useState<number | null>(null);
  const [isUploadingMedia, setIsUploadingMedia] = useState(false);
  const [mediaConfig, setMediaConfig] = useState({
    forced_language: '',
    chunk_min_duration: 30,
    chunk_max_duration: 120,
    chunk_overlap: 5
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load AI services for transcription
  useEffect(() => {
    const loadAIServices = async () => {
      try {
        const services = await apiService.getAIServices(appId);
        setAiServices(services || []);
      } catch (error) {
        console.error('Error loading AI services:', error);
      }
    };
    loadAIServices();
  }, [appId]);

  // Update current conversation ID when prop changes
  useEffect(() => {
    setCurrentConversationId(conversationId || null);
  }, [conversationId]);

  // Load conversation history and persistent files on mount or when conversation changes
  useEffect(() => {
    const loadConversationHistory = async () => {
      try {
        setIsLoadingHistory(true);
        
        // If we have a specific conversation ID, load from that conversation
        if (currentConversationId) {
          const response = await apiService.getConversationWithHistory(currentConversationId);
          
          if (response.messages && response.messages.length > 0) {
            const loadedMessages: Message[] = response.messages.map((msg: any, index: number) => ({
              id: `history-${index}`,
              type: msg.role === 'user' ? 'user' : 'agent',
              content: msg.content,
              timestamp: new Date(),
            }));
            
            setMessages(loadedMessages);
            console.log(`Loaded ${loadedMessages.length} messages from conversation ${currentConversationId}`);
          } else {
            setMessages([]);
          }
        } else {
          // Fallback to old method for backward compatibility
          const response = await apiService.getConversationHistory(appId, agentId);
          
          if (response.messages && response.messages.length > 0) {
            const loadedMessages: Message[] = response.messages.map((msg: any, index: number) => ({
              id: `history-${index}`,
              type: msg.role === 'user' ? 'user' : 'agent',
              content: msg.content,
              timestamp: new Date(),
            }));
            
            setMessages(loadedMessages);
            console.log(`Loaded ${loadedMessages.length} messages from conversation history`);
          } else {
            setMessages([]);
          }
        }
      } catch (error) {
        console.error('Error loading conversation history:', error);
        setMessages([]);
      } finally {
        setIsLoadingHistory(false);
      }
    };

    const loadPersistentFiles = async () => {
      try {
        // Load files for the specific conversation (if any)
        const response = await apiService.listAttachedFiles(appId, agentId, currentConversationId);
        console.log('Persistent files response:', response);
        setPersistentFiles(response.files || []);
        console.log(`Loaded ${response.files?.length || 0} persistent files for conversation ${currentConversationId}:`, response.files);
      } catch (error) {
        console.error('Error loading persistent files:', error);
        setPersistentFiles([]);
      }
    };

    loadConversationHistory();
    loadPersistentFiles();
  }, [appId, agentId, currentConversationId]);

  useEffect(() => {
    if ((!metadataFields || metadataFields.length === 0) && filterMetadata !== undefined) {
      setFilterMetadata(undefined);
      setFiltersKey((prev) => prev + 1);
    }
  }, [metadataFields, filterMetadata]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() && persistentFiles.length === 0) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date(),
      files: persistentFiles.map(f => f.filename)
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const hasFilters = filterMetadata !== undefined && Object.keys(filterMetadata).length > 0;
      const searchParams = hasFilters ? filterMetadata : undefined;

      // Send message (files are already attached and will be included automatically)
      const response = await apiService.chatWithAgent(
        appId,
        agentId,
        inputMessage,
        [], // No new files with message - all files are pre-uploaded
        searchParams,
        currentConversationId
      );

      // Handle both string and JSON responses
      let responseContent = response.response || 'No response received';
      
      // If response is an object, convert to formatted JSON string
      if (typeof responseContent === 'object') {
        responseContent = JSON.stringify(responseContent, null, 2);
      }
      
      const agentMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'agent',
        content: responseContent,
        timestamp: new Date(),
        videoReferences: response.video_references || undefined
      };

      setMessages(prev => [...prev, agentMessage]);
      
      // If backend returned a conversation_id and we don't have one yet, use it
      if (response.conversation_id && !currentConversationId) {
        setCurrentConversationId(response.conversation_id);
        if (onConversationCreated) {
          onConversationCreated(response.conversation_id);
        }
      }
      
      // Notify parent component that a message was sent (to reload conversation list)
      if (onMessageSent) {
        onMessageSent();
      }
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'error',
        content: error instanceof Error ? error.message : 'An error occurred',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetConversation = async () => {
    try {
      // Stop all video polling before reset to prevent race conditions
      pollingIntervalsRef.current.forEach(interval => clearInterval(interval));
      pollingIntervalsRef.current.clear();
      setProcessingVideoFiles(new Set());
      
      // Pass conversation_id to properly clean up attached files for this specific conversation
      await apiService.resetAgentConversation(appId, agentId, currentConversationId || undefined);
      setMessages([]);
      setPersistentFiles([]);
      setFilterMetadata(undefined);
      setFiltersKey((prev) => prev + 1);
    } catch (error) {
      console.error('Error resetting conversation:', error);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    console.log('Uploading files:', files.map(f => f.name));
    
    setIsLoadingFiles(true);
    
    // Ensure we have a conversation to attach files to
    // This prevents files from being uploaded to a "global" session that gets lost
    let targetConversationId = currentConversationId;
    
    if (!targetConversationId) {
      try {
        // Create a conversation before uploading files
        console.log('No conversation exists, creating one for file attachment...');
        const convResponse = await apiService.createConversation(agentId);
        targetConversationId = convResponse.conversation_id;
        setCurrentConversationId(targetConversationId);
        
        // Notify parent component about the new conversation
        if (onConversationCreated && targetConversationId) {
          onConversationCreated(targetConversationId);
        }
        console.log(`Created conversation ${targetConversationId} for file attachment`);
      } catch (convError) {
        console.error('Error creating conversation for file upload:', convError);
        setIsLoadingFiles(false);
        event.target.value = '';
        return;
      }
    }
    
    // Upload files to persistent storage, associated with the conversation
    for (const file of files) {
      try {
        const uploadResponse = await apiService.uploadFileForChat(appId, agentId, file, targetConversationId);
        console.log(`Uploaded file: ${file.name} for conversation ${targetConversationId}`, uploadResponse);
      } catch (error) {
        console.error(`Error uploading file ${file.name}:`, error);
      }
    }
    
    // Reload persistent files for current conversation
    try {
      const response = await apiService.listAttachedFiles(appId, agentId, targetConversationId);
      console.log('Reloaded persistent files:', response);
      setPersistentFiles(response.files || []);
    } catch (error) {
      console.error('Error reloading persistent files:', error);
    } finally {
      setIsLoadingFiles(false);
    }
    
    // Clear the file input
    event.target.value = '';
  };

  const handleMediaUpload = async () => {
    // Need either files or YouTube URL
    if (mediaFiles.length === 0 && !youtubeUrl.trim()) return;
    
    setIsUploadingMedia(true);
    
    // Ensure we have a conversation
    let targetConversationId = currentConversationId;
    
    if (!targetConversationId) {
      try {
        console.log('No conversation exists, creating one for media upload...');
        const convResponse = await apiService.createConversation(agentId);
        targetConversationId = convResponse.conversation_id;
        setCurrentConversationId(targetConversationId);
        
        if (onConversationCreated && targetConversationId) {
          onConversationCreated(targetConversationId);
        }
        console.log(`Created conversation ${targetConversationId} for media upload`);
      } catch (convError) {
        console.error('Error creating conversation for media upload:', convError);
        setIsUploadingMedia(false);
        return;
      }
    }
    
    // Upload each media file
    for (const file of mediaFiles) {
      try {
        const uploadResponse = await apiService.uploadFileForChat(appId, agentId, file, targetConversationId);
        console.log(`Uploaded media: ${file.name}`, uploadResponse);
      } catch (error) {
        console.error(`Error uploading media ${file.name}:`, error);
      }
    }
    
    // Add YouTube video if URL provided
    if (youtubeUrl.trim()) {
      try {
        const youtubeResponse = await apiService.addYouTubeForChat(appId, agentId, youtubeUrl.trim(), targetConversationId, {
          forced_language: mediaConfig.forced_language || undefined,
          chunk_min_duration: mediaConfig.chunk_min_duration,
          chunk_max_duration: mediaConfig.chunk_max_duration,
          chunk_overlap: mediaConfig.chunk_overlap
        });
        console.log(`Added YouTube video: ${youtubeUrl}`, youtubeResponse);
        
        // Start polling for the YouTube video processing
        if (youtubeResponse.file_id) {
          startVideoPolling(youtubeResponse.file_id);
        }
      } catch (error) {
        console.error(`Error adding YouTube video ${youtubeUrl}:`, error);
        alert('Error adding YouTube video. Please check the URL and try again.');
      }
    }
    
    // Reload persistent files
    try {
      const response = await apiService.listAttachedFiles(appId, agentId, targetConversationId);
      setPersistentFiles(response.files || []);
    } catch (error) {
      console.error('Error reloading persistent files:', error);
    } finally {
      setIsUploadingMedia(false);
    }
    
    // Close modal and reset state
    setShowMediaUploadModal(false);
    setMediaFiles([]);
    setYoutubeUrl('');
  };

  const handleRemovePersistentFile = async (fileId: string) => {
    try {
      await apiService.removeAttachedFile(appId, agentId, fileId, currentConversationId);
      console.log(`Removed persistent file: ${fileId} from conversation ${currentConversationId}`);
      
      // Reload persistent files for current conversation
      const response = await apiService.listAttachedFiles(appId, agentId, currentConversationId);
      setPersistentFiles(response.files || []);
    } catch (error) {
      console.error(`Error removing file ${fileId}:`, error);
    }
  };

  // State for video processing
  const [processingVideoFiles, setProcessingVideoFiles] = useState<Set<string>>(new Set());
  const pollingIntervalsRef = useRef<Map<string, NodeJS.Timeout>>(new Map());

  // Start polling for video processing status
  const startVideoPolling = (fileId: string) => {
    if (pollingIntervalsRef.current.has(fileId)) return;
    
    const interval = setInterval(async () => {
      try {
        const response = await apiService.listAttachedFiles(appId, agentId, currentConversationId);
        const files = response.files || [];
        const file = files.find((f: { file_id: string }) => f.file_id === fileId);
        
        if (file) {
          setPersistentFiles(files);
          
          // Stop polling when processing is complete
          if (file.processing_status === 'ready' || file.processing_status === 'error') {
            stopVideoPolling(fileId);
            setProcessingVideoFiles(prev => {
              const newSet = new Set(prev);
              newSet.delete(fileId);
              return newSet;
            });
            console.log(`Video ${fileId} processing finished: ${file.processing_status}`);
          }
        }
      } catch (err) {
        console.error('Video polling error:', err);
        stopVideoPolling(fileId);
      }
    }, 2000); // Poll every 2 seconds
    
    pollingIntervalsRef.current.set(fileId, interval);
  };

  const stopVideoPolling = (fileId: string) => {
    const interval = pollingIntervalsRef.current.get(fileId);
    if (interval) {
      clearInterval(interval);
      pollingIntervalsRef.current.delete(fileId);
    }
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      pollingIntervalsRef.current.forEach(interval => clearInterval(interval));
    };
  }, []);

  const handleProcessVideo = async (fileId: string) => {
    try {
      // Mark as processing
      setProcessingVideoFiles(prev => new Set(prev).add(fileId));
      
      console.log(`Starting video processing: ${fileId} with config:`, mediaConfig);
      const result = await apiService.processAttachedVideo(appId, agentId, fileId, currentConversationId, {
        forced_language: mediaConfig.forced_language || undefined,
        chunk_min_duration: mediaConfig.chunk_min_duration,
        chunk_max_duration: mediaConfig.chunk_max_duration,
        chunk_overlap: mediaConfig.chunk_overlap
      });
      console.log('Video processing started:', result);
      
      // Update file status to processing immediately
      setPersistentFiles(prev => prev.map(f => 
        f.file_id === fileId ? { ...f, processing_status: 'processing' } : f
      ));
      
      // Start polling for completion
      startVideoPolling(fileId);
      
    } catch (error) {
      console.error(`Error starting video processing ${fileId}:`, error);
      alert('Error processing video. Check console for details.');
      // Remove from processing set on error
      setProcessingVideoFiles(prev => {
        const newSet = new Set(prev);
        newSet.delete(fileId);
        return newSet;
      });
    }
  };

  const handleFilterMetadataChange = useCallback((metadata: Record<string, unknown> | undefined) => {
    setFilterMetadata(metadata);
  }, []);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="space-y-6">
      {/* Metadata Filters Section */}
      {metadataFields && metadataFields.length > 0 && (
        <div className="bg-white shadow rounded-lg">
          <button
            type="button"
            className="w-full p-4 border-b flex items-center justify-between text-left hover:bg-gray-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            onClick={() => setIsFilterExpanded((prev) => !prev)}
            aria-expanded={isFilterExpanded}
            aria-controls={filterPanelId}
          >
            <h3 className="text-lg font-medium text-gray-900 flex items-center">
              <span className="mr-2" aria-hidden="true">🔍</span>{' '}
              Filter by Metadata
            </h3>
            <svg
              className={`w-5 h-5 text-gray-500 transform transition-transform ${isFilterExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          <div id={filterPanelId} className={`p-4 bg-gray-50 ${isFilterExpanded ? '' : 'hidden'}`}>
            <SearchFilters
              key={filtersKey}
              metadataFields={metadataFields}
              dbType={vectorDbType?.toUpperCase()}
              disabled={isLoading}
              onFilterMetadataChange={handleFilterMetadataChange}
            />
          </div>
        </div>
      )}

      {/* Chat Interface */}
      <div className="bg-white shadow rounded-lg">
        <div className="p-4 border-b">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900">
              <span className="mr-2">💬</span>
              Chat with {agentName}
            </h3>
            <button
              onClick={handleResetConversation}
              className="px-3 py-1 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              Reset Conversation
            </button>
          </div>
        </div>

        {/* Messages Container */}
        <div className="h-96 overflow-y-auto p-4 space-y-4">
          {isLoadingHistory ? (
            <div className="flex justify-center items-center h-full">
              <div className="text-gray-500">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-600 mx-auto mb-2"></div>
                Loading conversation...
              </div>
            </div>
          ) : (
            <>
              {messages.map((message) => {
                const isUserMessage = message.type === 'user';
                const isErrorMessage = message.type === 'error';
                const alignmentClass = isUserMessage ? 'justify-end' : 'justify-start';

                let bubbleClass = 'bg-gray-200 text-gray-900';
                let senderLabel = agentName;

                if (isUserMessage) {
                  bubbleClass = 'bg-blue-600 text-white';
                  senderLabel = 'You';
                } else if (isErrorMessage) {
                  bubbleClass = 'bg-red-600 text-white';
                  senderLabel = 'Error';
                }

                return (
                  <div key={message.id} className={`flex flex-col ${alignmentClass}`}>
                    <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${bubbleClass}`}>
                      <div className="text-sm font-medium mb-1">{senderLabel}</div>
                      <div>
                        <MessageContent content={message.content} />
                      </div>
                      {message.files && message.files.length > 0 && (
                        <div className="mt-2 text-xs opacity-75">
                          📎 {message.files.join(', ')}
                        </div>
                      )}
                      <div className="text-xs opacity-75 mt-1">
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                    {/* Video Player for relevant segments */}
                    {message.videoReferences && message.videoReferences.length > 0 && (
                      <div className="mt-2 max-w-2xl">
                        <VideoPlayer videoReferences={message.videoReferences} />
                      </div>
                    )}
                  </div>
                );
              })}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-200 text-gray-900 px-4 py-2 rounded-lg">
                    <div className="flex items-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-2"></div>
                      Thinking...
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 border-t bg-gray-50">
          {/* File Upload */}
          <div className="mb-3 flex gap-2">
            <input
              type="file"
              multiple
              onChange={handleFileUpload}
              className="hidden"
              id="file-upload"
              accept=".pdf,.txt,.md,.png,.jpg,.jpeg,.doc,.docx"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer inline-flex items-center px-3 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
            >
              📎 Attach File
            </label>
            
            {/* Media Upload */}
            <button
              onClick={() => setShowMediaUploadModal(true)}
              className="inline-flex items-center px-3 py-2 border border-purple-300 rounded-lg text-sm font-medium text-purple-700 bg-purple-50 hover:bg-purple-100"
            >
              🎥 Upload Video/Audio
            </button>
          </div>

          {/* Persistent Files with Visual Feedback */}
          {(persistentFiles.length > 0 || isLoadingFiles) && (
            <div className="mb-3">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm font-medium text-gray-700">Attached Files:</div>
                {persistentFiles.length > 0 && (
                  <span className="text-xs text-gray-500">
                    {persistentFiles.length} file{persistentFiles.length !== 1 ? 's' : ''}
                  </span>
                )}
              </div>
              {isLoadingFiles && (
                <div className="flex items-center justify-center py-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                  <span className="text-sm text-gray-500">Uploading files...</span>
                </div>
              )}
              <div className="space-y-2">
                {persistentFiles.map((file) => (
                  <div key={file.file_id} className="flex items-center justify-between bg-white px-3 py-2 rounded border hover:border-gray-400 transition-colors">
                    <div className="flex items-center space-x-3 flex-1 min-w-0">
                      {/* File Type Icon */}
                      <span className="text-lg flex-shrink-0">
                        {file.file_type === 'pdf' && '📄'}
                        {file.file_type === 'image' && '🖼️'}
                        {file.file_type === 'text' && '📝'}
                        {file.file_type === 'document' && '📑'}
                        {file.file_type === 'video' && (file.filename?.includes('YouTube') ? '📺' : '🎥')}
                        {file.file_type === 'audio' && '🎵'}
                        {!['pdf', 'image', 'text', 'document', 'video', 'audio'].includes(file.file_type) && '📁'}
                      </span>
                      
                      {/* File Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <span className="text-sm text-gray-800 font-medium truncate" title={file.filename}>
                            {file.filename}
                          </span>
                          {/* Processing Status Badge */}
                          {file.processing_status && (
                            <span className={`text-xs px-1.5 py-0.5 rounded-full flex-shrink-0 ${
                              file.processing_status === 'ready' 
                                ? 'bg-green-100 text-green-700' 
                                : file.processing_status === 'error' 
                                  ? 'bg-red-100 text-red-700'
                                  : 'bg-yellow-100 text-yellow-700'
                            }`}>
                              {file.processing_status === 'ready' && '✓ Ready'}
                              {file.processing_status === 'error' && '✗ Error'}
                              {file.processing_status === 'uploaded' && '⏳ Uploaded'}
                              {file.processing_status === 'processing' && '⏳ Processing'}
                              {file.processing_status === 'transcribing' && '🎙️ Transcribing'}
                              {file.processing_status === 'indexing' && '📑 Indexing'}
                              {file.processing_status === 'pending' && '⏳ Pending'}
                              {file.processing_status === 'downloading' && '⬇️ Downloading'}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center space-x-2 text-xs text-gray-500">
                          {/* File Size - don't show 0B for YouTube videos */}
                          {file.file_size_display && file.file_size_display !== '0B' && file.file_size_display !== '0 B' && (
                            <span>{file.file_size_display}</span>
                          )}
                          {/* Content Extraction Status - only show for non-video/audio files or when ready */}
                          {file.has_extractable_content !== undefined && (
                            <span className={file.has_extractable_content || file.file_type === 'image' ? 'text-green-600' : 'text-gray-400'}>
                              {file.file_type === 'image' 
                                ? '• Vision ready' 
                                : (file.file_type === 'video' || file.file_type === 'audio')
                                  ? file.processing_status === 'ready' 
                                    ? '• Transcribed'
                                    : '' // Don't show redundant status text - the badge already shows the status
                                  : file.has_extractable_content 
                                    ? '• Text extracted' 
                                    : '• No text'}
                            </span>
                          )}
                        </div>
                        {/* Content Preview */}
                        {file.content_preview && (
                          <div className="text-xs text-gray-400 truncate mt-1" title={file.content_preview}>
                            {file.content_preview}
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Action Buttons */}
                    <div className="flex items-center space-x-1 ml-2 flex-shrink-0">
                      {/* Process Video/Audio Button - show for unprocessed media that are not downloading or already processing */}
                      {(file.file_type === 'video' || file.file_type === 'audio') && 
                       file.processing_status !== 'ready' && 
                       file.processing_status !== 'downloading' &&
                       file.processing_status !== 'processing' &&
                       file.processing_status !== 'transcribing' &&
                       file.processing_status !== 'indexing' && (
                        <button
                          onClick={() => handleProcessVideo(file.file_id)}
                          disabled={processingVideoFiles.has(file.file_id)}
                          className={`text-xs px-2 py-1 rounded font-medium ${
                            processingVideoFiles.has(file.file_id)
                              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                              : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                          }`}
                          title="Process media: transcribe and create searchable chunks"
                        >
                          {processingVideoFiles.has(file.file_id) ? '⏳ Processing...' : '🔄 Process'}
                        </button>
                      )}
                      
                      {/* Remove Button */}
                      <button
                        onClick={() => handleRemovePersistentFile(file.file_id)}
                        className="text-red-600 hover:text-red-800 text-sm p-1 rounded hover:bg-red-50"
                        title="Remove file"
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Message Input */}
          <div className="flex space-x-2">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message here... (Enter to send, Shift+Enter for new line)"
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={3}
            />
            <button
              onClick={handleSendMessage}
              disabled={
                isLoading || 
                (!inputMessage.trim() && persistentFiles.length === 0) ||
                persistentFiles.some(f => (f.file_type === 'video' || f.file_type === 'audio') && f.processing_status !== 'ready')
              }
              title={
                persistentFiles.some(f => (f.file_type === 'video' || f.file_type === 'audio') && f.processing_status !== 'ready')
                  ? 'Process attached media before sending'
                  : undefined
              }
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </div>
        </div>
      </div>

      {/* Media Upload Modal */}
      <Modal
        isOpen={showMediaUploadModal}
        onClose={() => {
          setShowMediaUploadModal(false);
          setMediaFiles([]);
          setYoutubeUrl('');
        }}
        title="Upload Video/Audio"
      >
        <div className="space-y-4">
          {/* File Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Video/Audio Files
            </label>
            <input
              type="file"
              multiple
              accept="video/*,audio/*"
              onChange={(e) => setMediaFiles(Array.from(e.target.files || []))}
              className="w-full"
              disabled={!!youtubeUrl.trim()}
            />
            {mediaFiles.length > 0 && (
              <p className="text-sm text-gray-600 mt-2">{mediaFiles.length} file(s) selected</p>
            )}
          </div>

          {/* OR Separator */}
          <div className="flex items-center gap-4">
            <div className="flex-1 border-t border-gray-300"></div>
            <span className="text-sm text-gray-500 font-medium">OR</span>
            <div className="flex-1 border-t border-gray-300"></div>
          </div>

          {/* YouTube URL */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              YouTube URL
            </label>
            <input
              type="url"
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
              disabled={mediaFiles.length > 0}
            />
            {youtubeUrl && (
              <p className="text-sm text-gray-500 mt-1">
                The video will be downloaded and transcribed automatically.
              </p>
            )}
          </div>

          {/* Configuration */}
          <div className="border-t pt-4">
            <h3 className="font-medium mb-3">Processing Options (for file uploads)</h3>
            
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Transcription Service</label>
                {aiServices.length === 0 ? (
                  <div className="text-sm text-yellow-600 bg-yellow-50 p-2 rounded">
                    Note: YouTube videos use the system's default transcription service.
                  </div>
                ) : (
                  <select
                    value={selectedTranscriptionServiceId || ''}
                    onChange={(e) => setSelectedTranscriptionServiceId(e.target.value ? parseInt(e.target.value) : null)}
                    className="w-full px-3 py-2 border rounded-md text-sm"
                  >
                    <option value="">-- Select a Transcription Service --</option>
                    {aiServices.map(service => (
                      <option key={service.service_id} value={service.service_id}>
                        {service.name}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              <div>
                <label className="block text-sm text-gray-700 mb-1">Language (optional)</label>
                <select
                  value={mediaConfig.forced_language}
                  onChange={(e) => setMediaConfig({...mediaConfig, forced_language: e.target.value})}
                  className="w-full px-3 py-2 border rounded-md text-sm"
                >
                  <option value="">Auto-detect</option>
                  <option value="es">Spanish</option>
                  <option value="en">English</option>
                  <option value="eu">Basque</option>
                  <option value="fr">French</option>
                </select>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="block text-xs text-gray-700 mb-1">Min Chunk (s)</label>
                  <input
                    type="number"
                    value={mediaConfig.chunk_min_duration}
                    onChange={(e) => setMediaConfig({...mediaConfig, chunk_min_duration: parseInt(e.target.value)})}
                    className="w-full px-2 py-1 border rounded text-sm"
                    min="10"
                    max="60"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-700 mb-1">Max Chunk (s)</label>
                  <input
                    type="number"
                    value={mediaConfig.chunk_max_duration}
                    onChange={(e) => setMediaConfig({...mediaConfig, chunk_max_duration: parseInt(e.target.value)})}
                    className="w-full px-2 py-1 border rounded text-sm"
                    min="60"
                    max="300"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-700 mb-1">Overlap (s)</label>
                  <input
                    type="number"
                    value={mediaConfig.chunk_overlap}
                    onChange={(e) => setMediaConfig({...mediaConfig, chunk_overlap: parseInt(e.target.value)})}
                    className="w-full px-2 py-1 border rounded text-sm"
                    min="0"
                    max="20"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4 border-t">
            <button
              onClick={() => {
                setShowMediaUploadModal(false);
                setYoutubeUrl('');
                setMediaFiles([]);
              }}
              className="px-4 py-2 text-gray-600 hover:text-gray-800"
            >
              Cancel
            </button>
            <button
              onClick={handleMediaUpload}
              disabled={
                (mediaFiles.length === 0 && !youtubeUrl.trim()) ||
                isUploadingMedia
              }
              className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50"
            >
              {isUploadingMedia ? 'Processing...' : youtubeUrl.trim() ? 'Add YouTube' : 'Upload'}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

export default ChatInterface; 