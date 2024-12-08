"use client";
import React, { useState } from 'react';


export default function Home() {

  return (
    <>
     <div className="flex justify-center items-center p-10 flex-col gap-2">
        <ImageUpload />
        <button className='btn btn-primary'>See Who looks like me</button>
      </div>
    </>
  );
}

function ImageUpload() {
  const [selectedImage, setSelectedImage] = useState(null);

  const handleImageChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedImage(URL.createObjectURL(event.target.files[0]));
    }
  };

  return (
    <div className="image-upload flex flex-col gap-2">
      <input type="file" accept="image/*" onChange={handleImageChange} />
      {selectedImage && <img src={selectedImage} alt="Selected"  className="uploaded-image w-80" />}
    </div>
  );
}

