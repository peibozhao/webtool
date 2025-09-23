
import { useState } from "react";
import { backendServer } from '../common/utils';
import s from "./SuperResolution.module.css"

function SuperResolution() {
  const [localImageFile, setLocalImageFile] = useState<{ location: string | null, file: File | null }>({ location: null, file: null });
  const [genImageUrl, setGenImageUrl] = useState<string | null>(null);

  const backend_server = backendServer();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    if (selectedFile) {
      setLocalImageFile({ location: URL.createObjectURL(selectedFile), file: selectedFile });
    }
  };

  const handleUpload = async () => {
    if (!localImageFile.file) return;

    const formData = new FormData();
    formData.append("image", localImageFile.file);
    try {
      const response = await fetch(`${backend_server}/api/super_resolution`, {
        method: "POST",
        body: formData,
      });
      const data = await response.blob();
      const file = new File([data], "times4.jpeg", { type: "image/jpeg" });
      const url = URL.createObjectURL(file);
      setGenImageUrl(url);
    } catch (error) {
      console.error("异常: ", error);
    }
  };

  return (
    <div className={s.root}>
      <div>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button onClick={handleUpload}>上传</button>
      </div>
      <div className={s.imageContainer}>
        {localImageFile.location && <img src={localImageFile.location} className={s.image} />}
        {genImageUrl && <img src={genImageUrl} className={s.image} />}
      </div>
    </div>
  );
}

export default SuperResolution;

