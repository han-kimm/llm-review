import ollama from "ollama";

async function test() {
  const res = await ollama.chat({
    model: "codeqwen",
    messages: [
      {
        role: "system",
        content: "You are a strict and perfect code reviewr AI.",
      },
      {
        role: "user",
        content: `다음 코드에서 잘못된 동작을 찾아주세요. 현재 코드에서 콘솔에 어떻게 출력되는지도 알려주세요.
            const test = async () => {
    const a = Array.from({length: 10}, (v, i) => Promise.resolve(i));
    a.forEach(async (v) => {
        console.log(v);
    } )
    
}`,
      },
    ],
  });

  console.log(res.message.content);
}

test();
