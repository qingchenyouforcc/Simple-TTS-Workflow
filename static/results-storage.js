// IndexedDB preserves uploaded reference audio as well as candidate metadata.
const resultStorage = (() => {
  let database;
  function open() {
    if (!database) database = new Promise((resolve, reject) => {
      const request = indexedDB.open("simple-tts-workspace", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("workspace");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error("本地存储被其他页面占用"));
    });
    return database;
  }
  async function access(mode, value) {
    const db = await open();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction("workspace", mode);
      const store = transaction.objectStore("workspace");
      const request = mode === "readonly" ? store.get("results") : store.put(value, "results");
      transaction.oncomplete = () => resolve(request.result);
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error || new Error("本地保存已中断"));
    });
  }
  return { load: () => access("readonly"), save: (value) => access("readwrite", value) };
})();
