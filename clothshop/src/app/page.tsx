import Image from 'next/image';
import Link from 'next/link';

export default function HomePage() {
	return (
		<div>
			<section className="relative">
				<div className="container py-12 md:py-20 grid gap-8 md:grid-cols-2 items-center">
					<div className="space-y-6">
						<h1 className="text-4xl md:text-6xl font-extrabold tracking-tight">Elevate Your Everyday</h1>
						<p className="text-neutral-600 text-lg max-w-prose">Premium apparel engineered for movement and comfort. Discover our latest drops across running, lifestyle, and training.</p>
						<div className="flex gap-3">
							<Link href="/men" className="bg-black text-white px-6 py-3 rounded-full">Shop Men</Link>
							<Link href="/women" className="bg-neutral-200 text-black px-6 py-3 rounded-full">Shop Women</Link>
						</div>
					</div>
					<div className="relative aspect-[4/3] md:aspect-[5/4]">
						<Image src="https://images.unsplash.com/photo-1519744792095-2f2205e87b6f?q=80&w=1600&auto=format&fit=crop" alt="Hero" fill className="object-cover rounded-3xl"/>
					</div>
				</div>
			</section>

			<section className="container py-12">
				<h2 className="text-2xl font-bold mb-6">Shop by Category</h2>
				<div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
					{[
						{ href: '/men', title: 'Men', img: 'https://images.unsplash.com/photo-1559563458-527698bf5295?q=80&w=1200&auto=format&fit=crop' },
						{ href: '/women', title: 'Women', img: 'https://images.unsplash.com/photo-1519741497674-611481863552?q=80&w=1200&auto=format&fit=crop' },
						{ href: '/kids', title: 'Kids', img: 'https://images.unsplash.com/photo-1605408499391-6368c628ef42?q=80&w=1200&auto=format&fit=crop' },
					].map((c) => (
						<Link key={c.title} href={c.href} className="group relative overflow-hidden rounded-2xl">
							<div className="relative aspect-[4/3]">
								<Image src={c.img} alt={c.title} fill className="object-cover transition-transform duration-500 group-hover:scale-105"/>
							</div>
							<div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent" />
							<div className="absolute bottom-4 left-4 text-white text-xl font-semibold">{c.title}</div>
						</Link>
					))}
				</div>
			</section>
		</div>
	);
}