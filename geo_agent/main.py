import os
import sys
import json
import random

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from geo_agent.agent.optimizer import GEOAgent
from geo_agent.core.models import WebPage

def main(task="optimize"):
    # 模拟输入
    html_dir = "cw22_search_dataset/data/html_samples"
    queries_dir = "query_generator/data/processed_dedup"
    loc_doc_path = "cw22_search_dataset/data/geo_bench_unique_docs.json"
    with open(loc_doc_path, "r", encoding="utf-8") as f:
        loc_docs = json.load(f)
    
    processed_html_files = [f for f in os.listdir(queries_dir) if f.endswith(".json")]
    for processed_html_file in processed_html_files[:4]:
        file_path = os.path.join(queries_dir, processed_html_file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        
        json_data = json.loads(data)
        train_queries = json_data.get("dataset", {}).get("train", [])
        test_queries = json_data.get("dataset", {}).get("test", [])


        file_uuid = os.path.splitext(processed_html_file)[0]
        target_doc = next((item for item in loc_docs if item["uuid"] == file_uuid), None)


        html_path = os.path.join(html_dir, file_uuid) + ".html"

        with open(html_path, "r", encoding="utf-8") as f:
            raw_html = f.read()
        
        page = WebPage(
            url=target_doc.get("url", "http://example.com"),
            raw_html=raw_html,
            cleaned_content="""### Steps to Become a Social Media Influencer (Including LinkedIn)\n\n1. **Choose Your Niche:** Identify a specific area you are passionate about and knowledgeable in to attract a dedicated audience.  \n2. **Select Your Main Platform:** Choose the platform that best fits your niche and target audience; LinkedIn is ideal for professional and industry-focused topics.  \n3. **Define Your Target Audience:** Understand your ideal followers' demographics, interests, and challenges to guide your content strategy.  \n4. **Create Consistent, Valuable Content:** Post regularly with high-quality, engaging content tailored to your audience using various formats like articles, videos, and images.  \n5. **Engage Authentically:** Respond to comments, participate in conversations, and interact with followers and other influencers to build community and loyalty.  \n6. **Collaborate with Other Influencers:** Partner with influencers in your niche through joint sessions, guest posts, or co-created content to expand your reach.  \n7. **Make It Easy for Brands to Contact You:** Clearly display contact information and prepare a professional media kit with audience metrics and collaboration options.  \n8. **Monetize Your Influence:** Explore revenue streams such as sponsored content, affiliate marketing, consulting, speaking engagements, and product sales.  \n9. **Continuously Learn and Adapt:** Stay updated on social media trends, platform changes, and audience preferences; use analytics to refine your strategy.\n\n---\n\n### LinkedIn Influencer Network Overview\n\nLinkedIn Influencers are recognized thought leaders and industry experts who share insights, trends, and advice focused on business and career development.\n\n**Key Aspects:**\n- **Selection and Recognition:** Influencers are typically invited based on expertise and achievements.  \n- **Content Focus:** Emphasis on professional development, industry trends, and leadership.  \n- **Community Engagement:** Influencers foster discussions, networking, and knowledge sharing.  \n- **Business Impact:** Collaborations with influencers can enhance brand credibility and expand reach.\n\n**Top LinkedIn Influencers by Expertise and Followers:**  \n- Richard Branson (Founder, Virgin Group) – 19.5M followers  \n- Bill Gates (Co-chair, Bill & Melinda Gates Foundation) – 34.4M followers  \n- Adam Grant (Organizational Psychologist, Author) – 4M+ followers  \n- Liz Ryan (Career Development Coach) – 3.1M followers  \n- Gary Vaynerchuk (Communications Expert, Author) – 4.1M followers  \n- Neil Patel (Co-Founder, Neil Patel Digital) – 417K followers  \n\n**How to Find and Engage with LinkedIn Influencers:**  \n- Use LinkedIn's advanced search filters by industry, location, and expertise.  \n- Join relevant LinkedIn Groups to connect and participate.  \n- Follow and engage authentically by liking, commenting, sharing, and tagging.  \n- Use LinkedIn Creator Mode to discover trending influencers.  \n- Attend LinkedIn Live events or webinars hosted by influencers.\n\n**Influencer Payment Models on LinkedIn:**  \n- LinkedIn does not directly pay influencers.  \n- Influencers monetize through sponsored content, consulting, speaking engagements, webinars, courses, and affiliate marketing.  \n- Collaborations typically involve negotiated fees or performance-based payments.\n\n**How to Work with LinkedIn Influencers:**  \n- Identify influencers aligned with your brand and goals.  \n- Build rapport through genuine engagement before proposing collaborations.  \n- Explore formats such as sponsored posts, webinars, LinkedIn Live sessions, or co-created articles.  \n- Define clear objectives, deliverables, and compensation upfront.\n\n**Common Questions:**  \n- **Do LinkedIn influencers get paid?**  \n  Not by LinkedIn directly; they monetize through external partnerships.  \n- **How do I tag influencers?**  \n  Use @ followed by their name in posts or comments.  \n- **Can anyone become a LinkedIn Influencer?**  \n  The official program is invitation-only, but many grow large followings independently.  \n- **What content performs best?**  \n  Professional insights, industry trends, leadership advice, authentic storytelling, and data-driven posts."""
            # HtmlParser().to_clean_text_bs4(raw_html)
        )
        
        agent = GEOAgent()
        if task == "optimize":
            random.shuffle(train_queries)
            optimized_page = agent.optimize_page(page, train_queries[:20])
            print("\nFinal Optimized Content Preview:")
            print(optimized_page.cleaned_content[:500])
        
        elif task == "evaluate":
            # 单独 evaluate 时，对原始 page 进行评估
            random.shuffle(test_queries)
            result = agent.evaluate_page(page, test_queries[:20])
            print(f"\nEvaluation Results for {file_uuid}: {result}")

        elif task == "both":
            # 先优化，再用 test 集评估优化后的页面
            random.shuffle(train_queries)
            optimized_page = agent.optimize_page(page, train_queries[:20])
            print("\nFinal Optimized Content Preview:")
            print(optimized_page.cleaned_content[:500])

            random.shuffle(test_queries)
            result = agent.evaluate_page(optimized_page, test_queries[:20])
            print(f"\nEvaluation Results for {file_uuid} (optimized): {result}")

        elif task == "compare":
            # 保存原始内容
            original_content = page.cleaned_content

            # 1. 评估原始页面 (baseline)
            random.shuffle(test_queries)
            test_subset = test_queries[:20]
            print("\n=== 评估原始页面 ===")
            baseline_result = agent.evaluate_page(page, test_subset)
            baseline_ratio = baseline_result['ratio']
            baseline_count = sum(v for k, v in baseline_result.items() if k != 'ratio')

            # 2. 优化页面
            print("\n=== 开始优化 ===")
            random.shuffle(train_queries)
            optimized_page = agent.optimize_page(page, train_queries[:20])

            # 3. 评估优化后页面 (用相同的 test_subset)
            print("\n=== 评估优化后页面 ===")
            optimized_result = agent.evaluate_page(optimized_page, test_subset)
            optimized_ratio = optimized_result['ratio']
            optimized_count = sum(v for k, v in optimized_result.items() if k != 'ratio')

            # 4. 输出对比结果
            total_queries = len(test_subset)
            print("\n" + "=" * 40)
            print("         优化效果对比")
            print("=" * 40)
            print(f"原始页面引用率:   {baseline_ratio:.1%} ({baseline_count}/{total_queries})")
            print(f"优化后引用率:     {optimized_ratio:.1%} ({optimized_count}/{total_queries})")
            print(f"提升幅度:         {optimized_ratio - baseline_ratio:+.1%} ({optimized_count - baseline_count:+d} queries)")
            print("=" * 40)

            # 5. 保存完整对比结果到 log
            compare_log = {
                "file_uuid": file_uuid,
                "url": page.url,
                "summary": {
                    "baseline_ratio": baseline_ratio,
                    "optimized_ratio": optimized_ratio,
                    "improvement": optimized_ratio - baseline_ratio,
                    "baseline_count": baseline_count,
                    "optimized_count": optimized_count,
                    "total_queries": total_queries
                },
                "content": {
                    "original": original_content,
                    "optimized": optimized_page.cleaned_content
                },
                "query_details": {
                    "baseline": {k: v for k, v in baseline_result.items() if k != 'ratio'},
                    "optimized": {k: v for k, v in optimized_result.items() if k != 'ratio'}
                }
            }

            compare_log_dir = "geo_agent/outputs/compare_logs"
            os.makedirs(compare_log_dir, exist_ok=True)
            compare_log_path = os.path.join(compare_log_dir, f"{file_uuid}_compare.json")
            with open(compare_log_path, 'w', encoding='utf-8') as f:
                json.dump(compare_log, f, indent=2, ensure_ascii=False)
            print(f"\n对比结果已保存至: {compare_log_path}")

if __name__ == "__main__":
    main(task="optimize")
