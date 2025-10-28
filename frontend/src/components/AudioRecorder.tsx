import { useState, useRef } from "react";
import s from "./AudioRecorder.module.css"

async function hasAudioInputDevice(): Promise<boolean> {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const hasMic = devices.some(
    d => d.kind === "audioinput" && d.deviceId !== "default" && d.deviceId !== "communications"
  );
  return hasMic;
}

function AudioRecorder() {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [audioURL, setAudioURL] = useState<string>("");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    const existInput = await hasAudioInputDevice();
    if (!existInput) {
      console.error("无法获取麦克风设备");
      return;
    }
    try {
      const stream: MediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);

      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      mediaRecorderRef.current.onstop = () => {
        mediaRecorderRef.current?.stream.getTracks().forEach(track => track.stop());
        const blob = new Blob(audioChunksRef.current, { type: mediaRecorderRef.current?.mimeType });
        const url = URL.createObjectURL(blob);
        setAudioURL(url);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      console.error("无法获取麦克风权限", err);
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  return (
    <div>
      <button onClick={isRecording ? stopRecording : startRecording}>
        {isRecording ? "停止录音" : "开始录音"}
      </button>

      {audioURL && (
        <div className={s.audioContainer}>
          <h4>录音播放：</h4>
          <audio src={audioURL} controls />
          <a href={audioURL} download="recording.wav">下载录音</a>
        </div>
      )}
    </div>
  );
};

export default AudioRecorder;

