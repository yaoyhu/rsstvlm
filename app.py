import asyncio
import base64
import io

import streamlit as st
from PIL import Image
from rsstvlm.agent.workflow import AgentWorkflow, StreamEvent
from rsstvlm.utils import deepseek_agent

st.set_page_config(
    page_title="天空地一体化超光谱遥感应用工程实验室",
    page_icon="🛰️",
    layout="wide",
)


# ======================
# 🤖 Agent 初始化 (cached)
# ======================
@st.cache_resource
def get_agent():
    """Initialize agent once and cache it."""
    return asyncio.run(
        AgentWorkflow.create(deepseek_agent, timeout=1200, verbose=True)
    )


# ======================
# 🧩 侧边栏功能
# ======================
with st.sidebar:
    st.header("🔧 控制面板")

    if st.button("🗑️ 清空当前对话"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # 加载 Agent 并显示工具
    with st.spinner("正在加载 Agent..."):
        try:
            agent = get_agent()
            st.success("Agent 已就绪 ✅")
            st.markdown("#### 🛠️ 可用工具")
            for tool in agent.tools:
                with st.expander(f"📦 {tool.metadata.name}"):
                    st.markdown(
                        f"**Description:** {tool.metadata.description}"
                    )
                    if tool.metadata.fn_schema:
                        st.markdown("**Args:**")
                        schema = tool.metadata.fn_schema.model_json_schema()
                        if "properties" in schema:
                            for param, info in schema["properties"].items():
                                param_type = info.get("type", "any")
                                param_desc = info.get("description", "")
                                st.markdown(
                                    f"- `{param}` ({param_type}): {param_desc}"
                                )
        except Exception as e:
            st.error(f"Agent 加载失败: {e}")

    st.divider()
    st.markdown("### 📌 使用说明")
    st.markdown("""
    - 目前工具较少，逐渐完善
    """)  # noqa: RUF001

# ======================
# 💬 聊天历史初始化
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list):
            # 多模态消息
            for part in message["content"]:
                if part["type"] == "text":
                    st.markdown(part["text"])
                elif part["type"] == "image_url":
                    st.image(part["image_url"]["url"], width=300)
        else:
            st.markdown(message["content"])

# ======================
# 🖼️ 图像上传 + 文本输入
# ======================
uploaded_file = st.file_uploader("📎 上传图像/视频")

# 检查是否已有太多消息(防过载)
if len(st.session_state.messages) >= 10:
    st.warning("对话较长，建议点击侧边栏「清空当前对话」以获得最佳体验。")  # noqa: RUF001

prompt = st.chat_input("请输入你的问题")


# ======================
# 🔄 Agent 调用函数
# ======================
def run_agent_with_placeholder(query: str, placeholder) -> tuple[str, list]:
    """Run agent with live streaming to a placeholder."""
    agent = get_agent()

    async def stream_to_placeholder():
        full_response = ""
        handler = agent.run(input=query)
        async for event in handler.stream_events():
            if isinstance(event, StreamEvent):
                full_response += event.delta
                placeholder.markdown(full_response + "▌")
        result = await handler  # 获取完整结果

        # 如果流式响应为空,从最终结果中获取响应内容
        if not full_response and result:
            response = result.get("response")
            if response and hasattr(response, "message"):
                full_response = str(response.message.content or "")

        sources = result.get("sources", []) if result else []
        return full_response, sources

    return asyncio.run(stream_to_placeholder())


# ======================
# 🤖 处理用户输入
# ======================
if prompt:
    # 构建用户内容
    user_content = [{"type": "text", "text": prompt}]

    # 如果上传了图片
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_url = f"data:image/jpeg;base64,{img_str}"
        user_content.insert(
            0, {"type": "image_url", "image_url": {"url": image_url}}
        )

    # 显示用户消息
    with st.chat_message("user"):
        if uploaded_file is not None:
            st.image(image, width=300)
        st.markdown(prompt)

    # 添加到历史
    st.session_state.messages.append({"role": "user", "content": user_content})

    # ======================
    # 🧠 调用 Agent
    # ======================
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []

        try:
            full_response, sources = run_agent_with_placeholder(
                prompt, message_placeholder
            )
            message_placeholder.markdown(full_response)

            # 显示数据来源
            if sources:
                with st.expander("📚 数据来源", expanded=False):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**{i}. {source.tool_name}**")
                        content = source.content
                        st.code(
                            content[:10000] + "..."
                            if len(content) > 10000
                            else content
                        )
        except Exception as e:
            error_msg = f"❌ 模型调用失败: {e!s}"
            message_placeholder.error(error_msg)
            full_response = error_msg

    # 添加助手回复
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
