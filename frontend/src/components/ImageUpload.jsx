import { useRef, useState } from "react";

export default function ImageUpload({ onFileSelected, previewUrl }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  function handleFiles(files) {
    if (files && files.length > 0) {
      onFileSelected(files[0]);
    }
  }

  return (
    // Exposed as a button: the real <input type="file"> is visually hidden, so
    // without a role/tabIndex and key handling there is no keyboard path to
    // uploading at all.
    <div
      className={`upload-dropzone${dragOver ? " drag-over" : ""}${previewUrl ? " has-preview" : ""}`}
      role="button"
      tabIndex={0}
      aria-label={previewUrl ? "Replace the selected slope photo" : "Upload a ground-level slope photo"}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
      />
      {previewUrl ? (
        <img src={previewUrl} alt="Uploaded slope" className="upload-preview" />
      ) : (
        <div className="upload-empty">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M4 16.5V19a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2.5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M7 9l5-5 5 5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M12 4v13" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <p>Drop a ground-level slope photo, click to upload, or paste</p>
        </div>
      )}
    </div>
  );
}
