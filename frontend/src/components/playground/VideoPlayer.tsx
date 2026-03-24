import React, { useRef, useState, useEffect } from 'react';
import { configService } from '../../core/ConfigService';

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

interface VideoPlayerProps {
  videoReferences: VideoReference[];
  onClose?: () => void;
}


const getFullVideoUrl = (relativeUrl: string): string => {
  if (!relativeUrl) return '';
  // If already absolute, return as-is
  if (relativeUrl.startsWith('http://') || relativeUrl.startsWith('https://')) {
    return relativeUrl;
  }
  // Get API base URL and append the relative path
  const baseUrl = configService.getApiBaseUrl();
  // Remove trailing slash from baseUrl if present
  const cleanBase = baseUrl.replace(/\/$/, '');
  // Ensure relativeUrl starts with /
  const cleanPath = relativeUrl.startsWith('/') ? relativeUrl : `/${relativeUrl}`;
  return `${cleanBase}${cleanPath}`;
};


const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoReferences, onClose }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentUrl, setCurrentUrl] = useState<string>('');
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isExpanded, setIsExpanded] = useState(false);

  // Get the first video URL (assuming all references are from the same video)
  useEffect(() => {
    if (videoReferences.length > 0 && !currentUrl) {
      const fullUrl = getFullVideoUrl(videoReferences[0].video_url);
      console.log('Setting video URL:', videoReferences[0].video_url, '->', fullUrl);
      setCurrentUrl(fullUrl);
    }
  }, [videoReferences, currentUrl]);

  // Update time display and handle video events
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => setCurrentTime(video.currentTime);
    const handleDurationChange = () => setDuration(video.duration);
    
    const handleError = (e: Event) => {
      console.error('Video load error:', e);
      console.error('Video src:', video.src);
      console.error('Video error code:', video.error?.code);
      console.error('Video error message:', video.error?.message);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('durationchange', handleDurationChange);
    video.addEventListener('error', handleError);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('durationchange', handleDurationChange);
      video.removeEventListener('error', handleError);
    };
  }, []);

  const handleTimestampClick = (ref: VideoReference, index: number) => {
    const video = videoRef.current;
    if (!video) return;
    
    console.log('Timestamp clicked:', { index, start_time: ref.start_time });
    setSelectedTimestamp(index);
    
    // Seek to the start time of the reference and play
    video.currentTime = ref.start_time;
    video.play().catch(e => console.log('Autoplay prevented:', e));
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Check if any references are agent-cited
  const hasAgentCitedClips = videoReferences.some(ref => ref.is_agent_cited);

  if (videoReferences.length === 0) {
    return null;
  }

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden shadow-lg">
      {/* Header */}
      <div 
        className="flex items-center justify-between px-4 py-2 bg-gray-800 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <span className="text-purple-400">{hasAgentCitedClips ? '📍' : '🎬'}</span>
          <span className="text-white text-sm font-medium">
            {hasAgentCitedClips 
              ? `Momentos Citados (${videoReferences.length})`
              : `Momentos Relevantes (${videoReferences.length})`
            }
          </span>
          <span className="text-gray-500 text-xs">
            - Haz clic para {isExpanded ? 'minimizar' : 'ver el video'}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={(e) => { e.stopPropagation(); setIsExpanded(!isExpanded); }}
            className="text-gray-400 hover:text-white p-1 rounded"
            title={isExpanded ? "Minimizar" : "Expandir"}
          >
            {isExpanded ? '▼' : '▲'}
          </button>
          {onClose && (
            <button
              onClick={(e) => { e.stopPropagation(); onClose(); }}
              className="text-gray-400 hover:text-white p-1 rounded"
              title="Cerrar"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="flex flex-col">
          {/* Full Video Player */}
          <div className="bg-black">
            {currentUrl ? (
              <video
                ref={videoRef}
                src={currentUrl}
                className="w-full aspect-video"
                controls
                preload="metadata"
              />
            ) : (
              <div className="w-full aspect-video flex items-center justify-center text-gray-500">
                <span>No hay video disponible</span>
              </div>
            )}
          </div>

          {/* Progress Bar - Visual indicator of current position */}
          {duration > 0 && (
            <div className="bg-gray-900 px-4 py-3">
              <div className="flex items-center gap-3">
                <span className="text-white font-mono text-sm min-w-[50px]">
                  {formatTime(currentTime)}
                </span>
                <div className="flex-1 h-6 bg-gray-700 rounded-lg relative overflow-visible">
                  {/* Progress fill - behind everything */}
                  <div 
                    className="absolute inset-y-0 left-0 bg-purple-600/50 rounded-lg transition-all duration-150"
                    style={{ width: `${(currentTime / duration) * 100}%` }}
                  />
                  
                  {/* Cited timestamp markers - show as prominent blocks */}
                  {videoReferences.map((ref, index) => {
                    const startPercent = Math.min((ref.start_time / duration) * 100, 100);
                    const endPercent = Math.min((ref.end_time / duration) * 100, 100);
                    const widthPercent = Math.max(endPercent - startPercent, 3); // Minimum 3% width
                    const isSelected = selectedTimestamp === index;
                    
                    return (
                      <div
                        key={`marker-${index}`}
                        className={`absolute top-0 bottom-0 cursor-pointer transition-all ${
                          isSelected
                            ? 'bg-green-400 shadow-lg shadow-green-500/50'
                            : ref.is_agent_cited 
                              ? 'bg-yellow-400 hover:bg-yellow-300 hover:shadow-lg hover:shadow-yellow-500/30' 
                              : 'bg-blue-400 hover:bg-blue-300 hover:shadow-lg hover:shadow-blue-500/30'
                        }`}
                        style={{ 
                          left: `${startPercent}%`, 
                          width: `${widthPercent}%`,
                          minWidth: '8px',
                          borderRadius: '4px',
                          zIndex: isSelected ? 15 : 10,
                        }}
                        onClick={() => handleTimestampClick(ref, index)}
                        title={`${ref.start_formatted} - ${ref.end_formatted}`}
                      />
                    );
                  })}
                  
                  {/* Playhead indicator */}
                  <div 
                    className="absolute top-1/2 -translate-y-1/2 w-5 h-5 bg-white rounded-full shadow-lg border-2 border-purple-500 transition-all duration-150 pointer-events-none"
                    style={{ 
                      left: `calc(${(currentTime / duration) * 100}% - 10px)`,
                      zIndex: 30,
                    }}
                  />
                </div>
                <span className="text-gray-400 font-mono text-sm min-w-[50px] text-right">
                  {formatTime(duration)}
                </span>
              </div>
              {/* Legend */}
              <div className="flex items-center gap-4 mt-2 text-xs text-gray-400">
                <div className="flex items-center gap-1">
                  <div className="w-4 h-3 bg-yellow-400 rounded"></div>
                  <span>Citado</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-4 h-3 bg-blue-400 rounded"></div>
                  <span>Relevante</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-4 h-3 bg-green-400 rounded"></div>
                  <span>Seleccionado</span>
                </div>
                <span className="text-gray-600">({videoReferences.length} momentos)</span>
              </div>
            </div>
          )}

          {/* Timestamp Navigation Buttons */}
          <div className="bg-gray-800 p-3">
            <div className="text-xs text-gray-500 mb-2">
              Salta a los momentos mencionados:
            </div>
            <div className="flex flex-wrap gap-2">
              {videoReferences.map((ref, index) => (
                <button
                  key={`${ref.file_id}-${index}`}
                  onClick={() => handleTimestampClick(ref, index)}
                  className={`
                    flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium
                    transition-all duration-200 
                    ${selectedTimestamp === index 
                      ? 'bg-purple-600 text-white ring-2 ring-purple-400' 
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:text-white'
                    }
                  `}
                  title={ref.text_preview}
                >
                  {ref.is_agent_cited && <span className="text-yellow-300">📍</span>}
                  <span className="font-mono">
                    {ref.start_formatted}
                  </span>
                  {ref.start_formatted !== ref.end_formatted && (
                    <span className="text-gray-400 font-mono">
                      - {ref.end_formatted}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Current timestamp info */}
          {selectedTimestamp !== null && videoReferences[selectedTimestamp] && (
            <div className="bg-gray-800/50 p-3 border-t border-gray-700">
              <p className="text-gray-300 text-sm">
                {videoReferences[selectedTimestamp].text_preview}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default VideoPlayer;
